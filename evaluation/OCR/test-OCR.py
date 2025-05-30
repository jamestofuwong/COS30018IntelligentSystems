import os
import csv
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import string
import numpy as np
from scipy.special import softmax
import collections
from pathlib import Path

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

class ImageOCRReader:
    
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
        
    def predict(self, img_path):
        """Predict from image file path"""
        # image processing for crnn
        image = Image.open(img_path)
        
        return crnn_predict(self.crnn, image, self.transformer, self.decoder)

def process_images_folder(folder_path, model_path, output_csv, decoder='bestPath'):
    """
    Process all images in a folder and save OCR results to CSV in alphabetical order
    
    Args:
        folder_path (str): Path to folder containing images
        model_path (str): Path to the CRNN model (.pth file)
        output_csv (str): Path to output CSV file
        decoder (str): 'bestPath' or 'beamSearch'
    """
    
    # Initialize the OCR reader
    print(f"Initializing OCR reader with decoder: {decoder}")
    ocr_reader = ImageOCRReader(decoder=decoder)
    
    try:
        ocr_reader.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Get list of image files
    folder_path = Path(folder_path)
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = [f for f in folder_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
    
    # Sort files alphabetically, accounting for numbers in the names
    image_files = sorted(image_files, key=lambda f: (f.name.lower()))
    
    print(f"Found {len(image_files)} image files")
    
    # Process images and collect results
    results = []
    
    for i, image_file in enumerate(image_files, 1):
        try:
            print(f"Processing {i}/{len(image_files)}: {image_file.name}")
            
            # Predict text from image
            predicted_text = ocr_reader.predict(str(image_file))
            
            # Append the result
            results.append({'filename': image_file.name, 'text': predicted_text})
        
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            results.append({'filename': image_file.name, 'text': 'ERROR'})
    
    # Save results to CSV in alphabetical order of filenames
    try:
        with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['filename', 'text'])
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        print(f"Results saved to {output_csv}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

def main():
    # Hardcoded configuration
    input_folder = './platerecognition-test-dataset/images'  # Folder containing images to process
    model_path = './reader.pth'  # Path to CRNN model
    output_csv = './OCR-results/custom-trained-results.csv'  # Output CSV file
    decoder = 'beamSearch'  # CTC decoder method: 'bestPath' or 'beamSearch'
    
    # Validate inputs
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' does not exist")
        return
    
    print(f"Input folder: {input_folder}")
    print(f"Model path: {model_path}")
    print(f"Output CSV: {output_csv}")
    print(f"Decoder: {decoder}")
    print("-" * 50)
    
    # Process images
    process_images_folder(input_folder, model_path, output_csv, decoder)

if __name__ == "__main__":
    main()