# train_enhanced.py
import argparse
import utils
import model.crnn as model

import random
import os
import csv
import time
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import string
import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from PIL import Image, ImageOps
  
class LPDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions 
    __len__ and __getitem__.
    """
    def __init__(self, path, cv_idx, transform):
        """
        Store the filenames of the jpgs to use. 
        Specifies transforms to apply on images.

        Args:
            path: (string) directory containing the dataset
            cv_idx: cross validation indices (training / validation sets)
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.path = path
        temp = os.listdir(path)
        self.dirs = [temp[i] for i in cv_idx]
    
        filenames = [os.path.splitext(directory)[0] for directory in self.dirs]
        
        self.labels = [file_name.split('_')[-1] for file_name in filenames]
        self.transform = transform

    def __len__(self):
        # return size of dataset
        return len(self.dirs)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. 
        Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: corresponding label of image
        """
        image = Image.open(os.path.join(self.path, self.dirs[idx]))  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]    
    
def trainBatch(net, criterion, optimizer, converter, data_iter, device):
    """
    Parameters
    ----------
    net: crnn neural network
    criterion: CTC loss function
    optimizer: optimizer
    converter: text converter
    data_iter: one batch of datasets
    device: cuda or cpu
    """
    try:
        data = next(data_iter)
    except StopIteration:
        return None
        
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    
    # Move data to device
    images = cpu_images.to(device)
    
    # Encode text labels
    text_encoded, length_encoded = converter.encode(cpu_texts)
    text_encoded = text_encoded.to(device)
    length_encoded = length_encoded.to(device)
    
    # Forward pass
    preds = net(images)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device)
    
    # Apply log_softmax for CTC loss
    preds = F.log_softmax(preds, dim=2)
    
    # Calculate loss
    cost = criterion(preds, text_encoded, preds_size, length_encoded)
    
    # Backward pass
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    return cost.item()

def validation(net, dataset, batch_size, num_workers, criterion, converter, 
               device, max_iter=100):
    """
    To compute the validation loss and accuracy from a given validation dataset
    
    net: neural network architecture
    dataset: validation set
    criterion: loss function
    max_iter: maximum number of mini_batches
    converter: convert text into tensor
    device: cuda or cpu
    
    return: validation loss, accuracy
    """
    
    # Set model to evaluation mode
    net.eval()
    data_loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_iter = iter(data_loader)

    loss_avg = utils.averager()
    
    # measure accuracy
    y_true = []
    y_pred = []
    
    max_iter = min(max_iter, len(data_loader))
    
    with torch.no_grad():
        for i in range(max_iter):
            try:
                data = next(val_iter)
            except StopIteration:
                break
                
            cpu_images, cpu_texts = data
            batch_size = cpu_images.size(0)
            
            # Move data to device
            images = cpu_images.to(device)
            
            # Encode text labels
            text_encoded, length_encoded = converter.encode(cpu_texts)
            text_encoded = text_encoded.to(device)
            length_encoded = length_encoded.to(device)
            
            # Forward pass
            preds = net(images)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size).to(device)
            
            # Apply log_softmax for CTC loss
            preds_log = F.log_softmax(preds, dim=2)
            
            # Calculate loss
            cost = criterion(preds_log, text_encoded, preds_size, length_encoded)
            loss_avg.add(cost.item())
            
            # Make predictions for accuracy calculation
            _, preds_max = preds.max(2)
            preds_max = preds_max.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds_max.cpu(), preds_size.cpu(), raw=False)
            
            for pred, target in zip(sim_preds, cpu_texts):
                y_true.append(target.upper())
                y_pred.append(pred.upper())
    
    accuracy = accuracy_score(y_true, y_pred)
    
    if len(y_true) > 0:
        print(f"Sample: Ground Truth: '{y_true[0]}' | Prediction: '{y_pred[0]}'")
    
    return loss_avg.val(), accuracy

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_metrics_to_csv(csv_file, epoch, iteration, train_loss, val_loss, val_accuracy, learning_rate, elapsed_time):
    """Save training metrics to CSV file"""
    file_exists = os.path.isfile(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        # Write header if file doesn't exist
        if not file_exists:
            writer.writerow(['Epoch', 'Iteration', 'Train_Loss', 'Val_Loss', 'Val_Accuracy', 'Learning_Rate', 'Elapsed_Time'])
        
        writer.writerow([epoch, iteration, train_loss, val_loss, val_accuracy, learning_rate, elapsed_time])

def run():
    #### argument parsing ####
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
    parser.add_argument('--epoch', type=int, default=100, help='training epochs')
    parser.add_argument('--dataPath', required=True, help='path to training dataset')
    parser.add_argument('--savePath', required=True, help='path to save trained weights')
    parser.add_argument('--preTrainedPath', type=str, default=None,
                        help='path to pre-trained weights (incremental learning)')
    parser.add_argument('--seed', type=int, default=8888, help='reproduce experiement')
    parser.add_argument('--worker', type=int, default=0,
                        help='number of cores for data loading')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--maxLength', type=int, default=9, 
                        help='maximum license plate character length in data')
    parser.add_argument('--csvFile', type=str, default='training_metrics.csv', 
                        help='CSV file to save training metrics')
    opt = parser.parse_args()
    print(opt)

    best_val_acc = 0.0  # Track best validation accuracy
    best_model_path = os.path.join(opt.savePath, 'best_model.pth')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    #### set up constants and experiment settings ####
    IMGH = 32
    IMGW = 100

    cudnn.benchmark = True

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(opt.seed)

    if not os.path.exists(opt.savePath):
        os.makedirs(opt.savePath)

    #### data preparation & loading ####
    train_transformer = transforms.Compose([
        transforms.Grayscale(),  
        transforms.Resize((IMGH, IMGW)),
        transforms.ToTensor()])

    # Get all image files and split into train/val
    all_files = os.listdir(opt.dataPath)
    n = range(len(all_files))
    train_idx, val_idx = train_test_split(n, train_size=0.8, test_size=0.2, 
                                          random_state=opt.seed)

    # Create datasets
    print("Loading training data...")
    train_dataset = LPDataset(opt.dataPath, train_idx, train_transformer)
    train_loader = DataLoader(train_dataset, batch_size=opt.batchSize, 
                              num_workers=opt.worker, shuffle=True, pin_memory=True)
    
    print("Loading validation data...")
    val_dataset = LPDataset(opt.dataPath, val_idx, train_transformer)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    #### setup crnn model hyperparameters ####
    classes = string.ascii_uppercase + string.digits
    nclass = len(classes) + 1  # +1 for CTC blank
    nc = 1  # number of channels (grayscale)

    # Create CRNN model
    crnn = model.CRNN(IMGH, nc, nclass, 256).to(device)
    print("Model loaded successfully")

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        crnn = torch.nn.DataParallel(crnn)

    # Load pre-trained weights if provided
    if opt.preTrainedPath is not None:    
        print(f"Loading pre-trained weights from {opt.preTrainedPath}")
        crnn.load_state_dict(torch.load(opt.preTrainedPath, map_location=device))
    else:
        crnn.apply(weights_init)

    #### Setup loss function and optimizer ####
    converter = utils.strLabelConverter(classes)
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True).to(device)
    
    # Use Adam optimizer with lower learning rate
    optimizer = optim.Adam(crnn.parameters(), lr=opt.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    #### training begins ####
    print("Starting training...")
    start_time = time.time()
    
    # Calculate print/save intervals
    total_iterations = len(train_loader)
    print_interval = max(1, total_iterations // 4)  # Print 4 times per epoch
    save_interval = max(1, total_iterations // 2)   # Save 2 times per epoch
    
    for epoch in range(opt.epoch):
        epoch_start_time = time.time()
        crnn.train()
        
        train_iter = iter(train_loader)
        loss_avg = utils.averager()
        
        for i in range(total_iterations):
            # Train one batch
            cost = trainBatch(crnn, criterion, optimizer, converter, train_iter, device)
            if cost is None:
                break
                
            loss_avg.add(cost)
            
            # Print training progress
            if (i + 1) % print_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"[Epoch {epoch+1}/{opt.epoch}] Iteration {i+1}/{total_iterations}, Loss: {loss_avg.val():.4f}, Time: {elapsed_time:.2f}s")

        # Validation at end of epoch
        val_loss, val_acc = validation(crnn, val_dataset, opt.batchSize, opt.worker, criterion, converter, device)

        # Log metrics
        elapsed_time = time.time() - epoch_start_time
        save_metrics_to_csv(opt.csvFile, epoch+1, i+1, loss_avg.val(), val_loss, val_acc, scheduler.get_last_lr()[0], elapsed_time)

        print(f"Epoch {epoch+1} Summary - Train Loss: {loss_avg.val():.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(crnn.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {val_acc:.4f}")

        scheduler.step()

        
    print("Training completed!")
    print(f"Total training time: {time.time() - start_time:.2f} seconds")
    print(f"Training metrics saved to: {opt.csvFile}")
                
if __name__ == '__main__':
    run()