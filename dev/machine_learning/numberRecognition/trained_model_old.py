import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import glob
from PIL import Image
# from DatasetBuilder import test_loader, train_loader, batch_size, classes
import torchvision.transforms as transforms
from DigitRecognitionTraining import Network
import glob
import sys
from pathlib import Path
ROOT = ".git"

isWindows = sys.platform.startswith('win')
pathChar = "/"
if (isWindows): pathChar = "\\"

def get_root():
    new_path = Path(__file__)
    for i in str(new_path).split(pathChar):
        if (len(glob.glob( str(new_path/ ROOT)))>0): break
        new_path = new_path.parent.absolute()
    return new_path

root = get_root()

def classify(weights_path , img):
    loaded_model = Network()
    weights = torch.load(weights_path)
    loaded_model.load_state_dict(weights)
    image_trans = transforms.Compose([
                    transforms.Resize((32, 32), Image.ANTIALIAS),
                    transforms.ToTensor()
                 ])
    loaded_model.eval()

    image = image_trans(img).float()
    readyimg = image.unsqueeze(0)
    output = loaded_model(readyimg)
    _, predicted = torch.max(output, 1)
    
    return predicted

def predict_num(img, image_path=None):
    if image_path is not None:
        img = Image.open(image_path)
    path = str(Path(glob.glob(str(root)+"/**/*.pth",recursive = True)[0]   ))
    print("Path to Wighters:\n\n\n",path)
    return classify(path , img)
    

if __name__ == "__main__":
    num = predict_num(img=None, image_path="./DigitsGreen/205.png")
    print('Predicted: ', num)
