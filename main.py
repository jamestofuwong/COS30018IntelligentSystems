import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import os
import time
from multiprocessing import Process, Queue
import numpy as np

# Import all your existing classes and functions
import cv2
from ultralytics import YOLO
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import torchvision.transforms as transforms
from PIL import Image
import string
from scipy.special import softmax
import os
import time
from multiprocessing import Process, Queue
import collections
import itertools

# [Include all your existing classes here - strLabelConverter, BeamEntry, BeamState, 
# BidirectionalLSTM, CRNN, AutoLPR, and all the worker functions]
# For brevity, I'm not repeating them all, but they should be included exactly as in your original code

class strLabelConverter(object):
    """Convert between str and label.
    NOTE:
        Insert `blank` to the alphabet for CTC.
    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, ignore_case=True):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        self.alphabet = alphabet + '-'  # for `-1` index

        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1

    def encode(self, text):
        """Support batch or single str.
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        if isinstance(text, str):
            text = [
                self.dict[char.lower() if self._ignore_case else char]
                for char in text
            ]
            length = [len(text)]
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, t, length, raw=False):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            if raw:
                return ''.join([self.alphabet[i - 1] for i in t])
            else:
                char_list = []
                for i in range(length):
                    if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                        char_list.append(self.alphabet[t[i] - 1])
                return ''.join(char_list)
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l], torch.IntTensor([l]), raw=raw))
                index += l
            return texts

def ctcBestPath(mat, classes):
    "implements best path decoding as shown by Graves (Dissertation, p63)"

    # dim0=t, dim1=c
    maxT, maxC = mat.shape
    label = ''
    blankIdx = len(classes)
    lastMaxIdx = maxC # init with invalid label

    for t in range(maxT):
        maxIdx = np.argmax(mat[t, :])

        if maxIdx != lastMaxIdx and maxIdx != blankIdx:
            label += classes[maxIdx]

        lastMaxIdx = maxIdx

    return label

class BeamEntry:
    "information about one single beam at specific time-step"
    def __init__(self):
        self.prTotal = 0 # blank and non-blank
        self.prNonBlank = 0 # non-blank
        self.prBlank = 0 # blank
        self.prText = 1 # LM score
        self.lmApplied = False # flag if LM was already applied to this beam
        self.labeling = () # beam-labeling

class BeamState:
    "information about the beams at specific time-step"
    def __init__(self):
        self.entries = {}

    def norm(self):
        "length-normalise LM score"
        for (k, _) in self.entries.items():
            labelingLen = len(self.entries[k].labeling)
            self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

    def sort(self):
        "return beam-labelings, sorted by probability"
        beams = [v for (_, v) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
        return [x.labeling for x in sortedBeams]

def addBeam(beamState, labeling):
    "add beam if it does not yet exist"
    if labeling not in beamState.entries:
        beamState.entries[labeling] = BeamEntry()

def ctcBeamSearch(mat, classes, lm, beamWidth=25):
    "beam search as described by the paper of Hwang et al. and the paper of Graves et al."

    blankIdx = len(classes)
    maxT, maxC = mat.shape

    # initialise beam state
    last = BeamState()
    labeling = ()
    last.entries[labeling] = BeamEntry()
    last.entries[labeling].prBlank = 1
    last.entries[labeling].prTotal = 1

    # go over all time-steps
    for t in range(maxT):
        curr = BeamState()

        # get beam-labelings of best beams
        bestLabelings = last.sort()[0:beamWidth]

        # go over best beams
        for labeling in bestLabelings:

            # probability of paths ending with a non-blank
            prNonBlank = 0
            # in case of non-empty beam
            if labeling:
                # probability of paths with repeated last char at the end
                prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

            # probability of paths ending with a blank
            prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

            # add beam at current time-step if needed
            addBeam(curr, labeling)

            # fill in data
            curr.entries[labeling].labeling = labeling
            curr.entries[labeling].prNonBlank += prNonBlank
            curr.entries[labeling].prBlank += prBlank
            curr.entries[labeling].prTotal += prBlank + prNonBlank
            curr.entries[labeling].prText = last.entries[labeling].prText
            curr.entries[labeling].lmApplied = True

            # extend current beam-labeling
            for c in range(maxC - 1):
                # add new char to current beam-labeling
                newLabeling = labeling + (c,)

                # if new labeling contains duplicate char at the end, only consider paths ending with a blank
                if labeling and labeling[-1] == c:
                    prNonBlank = mat[t, c] * last.entries[labeling].prBlank
                else:
                    prNonBlank = mat[t, c] * last.entries[labeling].prTotal

                # add beam at current time-step if needed
                addBeam(curr, newLabeling)
                
                # fill in data
                curr.entries[newLabeling].labeling = newLabeling
                curr.entries[newLabeling].prNonBlank += prNonBlank
                curr.entries[newLabeling].prTotal += prNonBlank

        # set new beam state
        last = curr

    # normalise LM scores according to beam-labeling-length
    last.norm()

      # sort by probability
    bestLabeling = last.sort()[0] # get most probable labeling

    # map labels to chars
    res = ''
    for l in bestLabeling:
        res += classes[l]

    return res

# CRNN Model Classes
class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)
        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'
        
        # kernel size
        ks = [3, 3, 3, 3, 3, 3, 2]
        # padding
        ps = [1, 1, 1, 1, 1, 1, 0]
        # stride
        ss = [1, 1, 1, 1, 1, 1, 1]
        # number of channels
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
                
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        cnn.add_module('dropout{0}'.format(2), nn.Dropout2d(p=0.2, inplace=False))
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16
        
        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        # batch size, channels, height, width
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(conv)
        return output

# CRNN Prediction Function
def crnn_predict(crnn, img, transformer, decoder='bestPath'):
    """
    Params
    ------
    crnn: torch.nn
        Neural network architecture
    transformer: torchvision.transform
        Image transformer
    decoder: string, 'bestPath' or 'beamSearch'
        CTC decoder method.
    
    Returns
    ------
    out: predicted alphanumeric sequence
    """
    
    classes = string.ascii_uppercase + string.digits
    image = img.copy()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    image = transformer(image).to(device)
    image = image.view(1, *image.size())
    
    # forward pass (convert to numpy array)
    preds_np = crnn(image).data.cpu().numpy().squeeze()
    
    # move first column to last (so that we can use CTCDecoder as it is)
    preds_np = np.hstack([preds_np[:, 1:], preds_np[:, [0]]])
    
    preds_sm = softmax(preds_np, axis=1)
                        
    if decoder == 'bestPath':
        output = ctcBestPath(preds_sm, classes)
        
    elif decoder == 'beamSearch':
        output = ctcBeamSearch(preds_sm, classes, None)
    else:
        raise Exception("Invalid decoder method. Choose either 'bestPath' or 'beamSearch'")
        
    return output

class AutoLPR:
    
    def __init__(self, decoder='bestPath'):
        
        # crnn parameters
        self.IMGH = 32
        self.nc = 1
        alphabet = string.ascii_uppercase + string.digits
        self.nclass = len(alphabet) + 1
        self.transformer = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize(self.IMGH),
            transforms.ToTensor()])
        self.decoder = decoder
                
    def load(self, crnn_path):
        # load CRNN
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.crnn = CRNN(self.IMGH, self.nc, self.nclass, nh=256).to(device)
        self.crnn.load_state_dict(torch.load(crnn_path, map_location=device))
            
        # remember to set to test mode (otherwise some layers might behave differently)
        self.crnn.eval()
        
    def predict_from_array(self, img_array):
        """Predict from numpy array (for video frames)"""
        # Convert numpy array to PIL Image
        if len(img_array.shape) == 3:
            # Convert BGR to RGB if needed
            if img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        
        image = Image.fromarray(img_array)
        
        return crnn_predict(self.crnn, image, self.transformer, self.decoder)
    
    def predict(self, img_path):
        """Predict from image file path"""
        # image processing for crnn
        image = Image.open(img_path)
        
        return crnn_predict(self.crnn, image, self.transformer, self.decoder)

def car_detection_worker(input_queue, car_output_queue, vehicle_model_path, vehicle_conf_threshold):
    vehicle_model = YOLO(vehicle_model_path)
    while True:
        frame_data = input_queue.get()
        if frame_data is None:  # Sentinel value to stop the worker
            car_output_queue.put(None)
            break

        frame_counter, frame = frame_data
        
        # Perform vehicle detection
        vehicle_results = vehicle_model(frame, conf=vehicle_conf_threshold, verbose=False)
        
        detections = {
            'vehicles': [],
            'plates': [],
            'frame_num': frame_counter,
            'original_frame': frame.copy() # Store original frame to pass to next stage
        }

        for result in vehicle_results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_conf = float(box.conf[0])
                
                detections['vehicles'].append({
                    'coords': (x1, y1, x2, y2),
                    'conf': vehicle_conf,
                    'plates': [] # This will be filled by the plate detection worker
                })
        
        car_output_queue.put(detections)

def plate_detection_worker(car_output_queue, plate_output_queue, plate_model_path, plate_conf_threshold):
    plate_model = YOLO(plate_model_path)
    while True:
        detections = car_output_queue.get()
        if detections is None: # Sentinel value
            plate_output_queue.put(None)
            break

        frame = detections['original_frame'] # Get the original frame

        for vehicle_data in detections['vehicles']:
            x1, y1, x2, y2 = vehicle_data['coords']
            vehicle_roi = frame[y1:y2, x1:x2]

            if vehicle_roi.size == 0:
                continue

            # License plate detection
            plate_results = plate_model(vehicle_roi, conf=plate_conf_threshold, verbose=False)

            for plate_result in plate_results:
                for plate_box in plate_result.boxes:
                    px1, py1, px2, py2 = map(int, plate_box.xyxy[0])
                    plate_conf = float(plate_box.conf[0])
                    
                    # Absolute coordinates
                    abs_px1 = x1 + px1
                    abs_py1 = y1 + py1
                    abs_px2 = x1 + px2
                    abs_py2 = y1 + py2
                    
                    # Store plate data
                    vehicle_data['plates'].append({
                        'coords': (abs_px1, abs_py1, abs_px2, abs_py2),
                        'conf': plate_conf,
                        'relative_roi': vehicle_roi[py1:py2, px1:px2].copy(), # Store plate ROI for OCR
                        'text': "" # This will be filled by the plate reading worker
                    })
        
        plate_output_queue.put(detections)

def plate_reading_worker(plate_output_queue, final_output_queue, crnn_model_path, ctc_decoder):
    lpr = AutoLPR(decoder=ctc_decoder)
    lpr.load(crnn_path=crnn_model_path)
    print("AutoLPR predictor initialized in plate reading worker.")

    while True:
        detections = plate_output_queue.get()
        if detections is None: # Sentinel value
            final_output_queue.put(None)
            break

        for vehicle_data in detections['vehicles']:
            for plate in vehicle_data['plates']:
                plate_roi = plate['relative_roi'] # Get the plate ROI

                if plate_roi.size > 0:
                    try:
                        plate_text = lpr.predict_from_array(plate_roi)
                        # Clean up the text (remove spaces and non-alphanumeric)
                        plate_text = ''.join(e for e in plate_text if e.isalnum()).upper()
                        plate['text'] = plate_text
                    except Exception as e:
                        print(f"AutoLPR prediction error in worker: {e}")
                        plate['text'] = ""
        
        final_output_queue.put(detections)

def is_image_file(file_path):
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')
    return file_path.lower().endswith(image_extensions)

class LicensePlateGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition System")
        self.root.geometry("900x700")
        
        # Configuration variables
        self.vehicle_model_path = './models/vehicle.pt'
        self.plate_model_path = './models/plate.pt'
        self.crnn_model_path = './models/reader.pth'
        self.vehicle_conf_threshold = 0.85
        self.plate_conf_threshold = 0.8
        self.ctc_decoder = 'beamSearch'
        
        # Current file variables
        self.current_file_path = None
        self.is_video = False
        self.detection_running = False
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="License Plate Recognition System", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Upload section
        upload_frame = ttk.LabelFrame(main_frame, text="Upload File", padding="10")
        upload_frame.grid(row=1, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        upload_frame.columnconfigure(1, weight=1)
        
        self.upload_button = ttk.Button(upload_frame, text="Upload Image/Video", 
                                       command=self.upload_file, width=20)
        self.upload_button.grid(row=0, column=0, padx=(0, 10))
        
        self.file_label = ttk.Label(upload_frame, text="No file selected", 
                                   foreground="gray")
        self.file_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        # Configuration section
        config_frame = ttk.LabelFrame(main_frame, text="Configuration", padding="10")
        config_frame.grid(row=1, column=3, sticky=(tk.W, tk.E, tk.N), padx=(10, 0), pady=(0, 10))
        
        ttk.Label(config_frame, text="Vehicle Confidence:").grid(row=0, column=0, sticky=tk.W)
        self.vehicle_conf_var = tk.DoubleVar(value=self.vehicle_conf_threshold)
        vehicle_conf_spinbox = ttk.Spinbox(config_frame, from_=0.1, to=1.0, increment=0.05,
                                          textvariable=self.vehicle_conf_var, width=10)
        vehicle_conf_spinbox.grid(row=0, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(config_frame, text="Plate Confidence:").grid(row=1, column=0, sticky=tk.W)
        self.plate_conf_var = tk.DoubleVar(value=self.plate_conf_threshold)
        plate_conf_spinbox = ttk.Spinbox(config_frame, from_=0.1, to=1.0, increment=0.05,
                                        textvariable=self.plate_conf_var, width=10)
        plate_conf_spinbox.grid(row=1, column=1, padx=(5, 0), pady=2)
        
        ttk.Label(config_frame, text="CTC Decoder:").grid(row=2, column=0, sticky=tk.W)
        self.decoder_var = tk.StringVar(value=self.ctc_decoder)
        decoder_combo = ttk.Combobox(config_frame, textvariable=self.decoder_var,
                                    values=['bestPath', 'beamSearch'], width=12)
        decoder_combo.grid(row=2, column=1, padx=(5, 0), pady=2)
        decoder_combo.state(['readonly'])
        
        # Preview and detection section
        preview_frame = ttk.LabelFrame(main_frame, text="Preview & Detection", padding="10")
        preview_frame.grid(row=2, column=0, columnspan=4, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        
        # Canvas for image/video display
        self.canvas = tk.Canvas(preview_frame, bg='lightgray', width=640, height=480)
        self.canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Scrollbars for canvas
        v_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.canvas.configure(yscrollcommand=v_scrollbar.set)
        
        h_scrollbar = ttk.Scrollbar(preview_frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.canvas.configure(xscrollcommand=h_scrollbar.set)
        
        # Control buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=(0, 10))
        
        self.detect_button = ttk.Button(button_frame, text="Start Detection", 
                                       command=self.start_detection, state='disabled')
        self.detect_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="Stop Detection", 
                                     command=self.stop_detection, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))    
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, 
                              relief=tk.SUNKEN, anchor=tk.W)
        status_bar.grid(row=4, column=0, columnspan=4, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Initialize canvas text
        self.canvas.create_text(320, 240, text="Upload an image or video to begin", 
                               fill="gray", font=('Arial', 12))
    
    def upload_file(self):
        file_types = [
            ('Image files', '*.png *.jpg *.jpeg *.bmp *.tiff *.webp'),
            ('Video files', '*.mp4 *.avi *.mov *.mkv *.wmv'),
            ('All files', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=file_types
        )
        
        if file_path:
            self.current_file_path = file_path
            self.is_video = not is_image_file(file_path)
            
            # Update UI
            filename = os.path.basename(file_path)
            file_type = "Video" if self.is_video else "Image"
            self.file_label.config(text=f"{file_type}: {filename}", foreground="black")
            self.detect_button.config(state='normal')
            
            # Show preview
            self.show_preview()
            self.status_var.set(f"Loaded {file_type.lower()}: {filename}")
    
    def show_preview(self):
        try:
            if self.is_video:
                # For video, show first frame
                cap = cv2.VideoCapture(self.current_file_path)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    self.display_frame(frame)
                else:
                    messagebox.showerror("Error", "Could not read video file")
            else:
                # For image
                frame = cv2.imread(self.current_file_path)
                if frame is not None:
                    self.display_frame(frame)
                else:
                    messagebox.showerror("Error", "Could not read image file")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {str(e)}")
    
    def display_frame(self, frame):
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not initialized yet, use default size
            canvas_width, canvas_height = 640, 480
        
        h, w = frame_rgb.shape[:2]
        scale = min(canvas_width/w, canvas_height/h)
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # Resize frame
        frame_resized = cv2.resize(frame_rgb, (new_width, new_height))
        
        # Convert to PIL Image and then to PhotoImage
        pil_image = Image.fromarray(frame_resized)
        self.photo = ImageTk.PhotoImage(pil_image)
        
        # Clear canvas and display image
        self.canvas.delete("all")
        self.canvas.create_image(canvas_width//2, canvas_height//2, image=self.photo)
        
        # Update canvas scroll region
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    
    def start_detection(self):
        if not self.current_file_path:
            messagebox.showwarning("Warning", "Please upload a file first")
            return
        
        # Check if model files exist
        required_models = [self.vehicle_model_path, self.plate_model_path, self.crnn_model_path]
        for model_path in required_models:
            if not os.path.exists(model_path):
                messagebox.showerror("Error", f"Model file not found: {model_path}")
                return
        
        # Update configuration from UI
        self.vehicle_conf_threshold = self.vehicle_conf_var.get()
        self.plate_conf_threshold = self.plate_conf_var.get()
        self.ctc_decoder = self.decoder_var.get()
        
        # Disable buttons and start detection
        self.detect_button.config(state='disabled')
        self.upload_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.detection_running = True
        
        # Start detection in separate thread
        self.detection_thread = threading.Thread(target=self.run_detection)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.status_var.set("Detection started...")
    
    def stop_detection(self):
        self.detection_running = False
        self.stop_button.config(state='disabled')
        self.status_var.set("Stopping detection...")
    
    def run_detection(self):
        try:
            # Create queues for inter-process communication
            raw_frames_queue = Queue(maxsize=10)
            car_detected_queue = Queue(maxsize=10)
            plate_detected_queue = Queue(maxsize=10)
            final_processed_queue = Queue(maxsize=10)

            frame_width, frame_height, fps = None, None, None

            if self.is_video:
                # Open video file
                cap = cv2.VideoCapture(self.current_file_path)
                if not cap.isOpened():
                    self.root.after(0, lambda: messagebox.showerror("Error", "Could not open video file"))
                    return

                # Get video properties
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
            else:
                # For image processing
                test_frame = cv2.imread(self.current_file_path)
                if test_frame is None:
                    self.root.after(0, lambda: messagebox.showerror("Error", "Could not read image file"))
                    return
                frame_height, frame_width = test_frame.shape[:2]

            # Start worker processes
            car_proc = Process(target=car_detection_worker, 
                             args=(raw_frames_queue, car_detected_queue, 
                                   self.vehicle_model_path, self.vehicle_conf_threshold))
            plate_proc = Process(target=plate_detection_worker, 
                               args=(car_detected_queue, plate_detected_queue, 
                                     self.plate_model_path, self.plate_conf_threshold))
            reader_proc = Process(target=plate_reading_worker, 
                                args=(plate_detected_queue, final_processed_queue, 
                                      self.crnn_model_path, self.ctc_decoder))

            car_proc.start()
            plate_proc.start()
            reader_proc.start()

            frame_counter = 0
            start_time = time.time()

            if self.is_video:
                # Process video frames
                while cap.isOpened() and self.detection_running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_counter += 1
                    raw_frames_queue.put((frame_counter, frame))
                    
                    # Check for processed frames to display
                    try:
                        if not final_processed_queue.empty():
                            detections = final_processed_queue.get_nowait()
                            if detections is not None:
                                processed_frame = self.draw_detections(detections)
                                self.root.after(0, lambda f=processed_frame: self.display_frame(f))
                    except:
                        pass
                    
                    # Small delay to prevent overwhelming the queue
                    time.sleep(0.03)  # ~30 FPS
                
                cap.release()
            else:
                # Process single image
                image_frame = cv2.imread(self.current_file_path)
                raw_frames_queue.put((1, image_frame))
                
                # Wait for processed result
                while self.detection_running:
                    try:
                        if not final_processed_queue.empty():
                            detections = final_processed_queue.get_nowait()
                            if detections is not None:
                                processed_frame = self.draw_detections(detections)
                                self.root.after(0, lambda f=processed_frame: self.display_frame(f))
                                break
                    except:
                        pass
                    time.sleep(0.1)

            # Signal workers to stop
            raw_frames_queue.put(None)
            
            # Wait for workers to finish
            car_proc.join(timeout=5)
            plate_proc.join(timeout=5)
            reader_proc.join(timeout=5)
            
            # Terminate processes if they're still running
            if car_proc.is_alive():
                car_proc.terminate()
            if plate_proc.is_alive():
                plate_proc.terminate()
            if reader_proc.is_alive():
                reader_proc.terminate()

            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Update UI in main thread
            self.root.after(0, self.detection_completed, elapsed_time)
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Detection error: {str(e)}"))
            self.root.after(0, self.detection_completed, 0)
    
    def draw_detections(self, detections):
        """Draw detection results on frame"""
        frame = detections['original_frame'].copy()
        
        for vehicle in detections.get('vehicles', []):
            x1, y1, x2, y2 = vehicle['coords']
            
            # Only draw vehicle box if it has plates
            if vehicle['plates']:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            for plate in vehicle['plates']:
                px1, py1, px2, py2 = plate['coords']
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 0, 255), 2)
                
                if plate['text']:
                    plate_label = f"{plate['text']}"
                    # Use bigger and thicker font
                    font_scale = 1.0
                    font_thickness = 2
                    (text_width, text_height), _ = cv2.getTextSize(plate_label, 
                                                                    cv2.FONT_HERSHEY_SIMPLEX, 
                                                                    font_scale, font_thickness)
                    
                    # Position text above the vehicle box with some padding
                    text_x = x1
                    text_y = y1 - 10
                    
                    # Make sure text doesn't go off screen
                    if text_y - text_height < 0:
                        text_y = y2 + text_height + 10
                    
                    # Draw black background rectangle for text
                    cv2.rectangle(frame, 
                                    (text_x, text_y - text_height - 5),
                                    (text_x + text_width + 10, text_y + 5),
                                    (0, 0, 0), -1)
                    
                    # Draw white text on black background
                    cv2.putText(frame, plate_label, 
                                (text_x + 5, text_y), 
                                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)
        
        return frame
    
    def detection_completed(self, elapsed_time):
        """Called when detection is completed"""
        self.detection_running = False
        self.detect_button.config(state='normal')
        self.upload_button.config(state='normal')
        self.stop_button.config(state='disabled')
        
        if elapsed_time > 0:
            self.status_var.set(f"Detection completed in {elapsed_time:.2f} seconds")
        else:
            self.status_var.set("Detection stopped")

def main():
    # Initialize multiprocessing support
    torch.multiprocessing.freeze_support()
    
    # Create and run the GUI
    root = tk.Tk()
    app = LicensePlateGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()