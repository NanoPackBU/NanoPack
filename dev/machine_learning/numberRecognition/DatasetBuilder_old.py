import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
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
## used for training ML 
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        img = Image.open(img_path)
        img = img.resize((32, 32), Image.ANTIALIAS)
        # image = torchvision.transforms.functional.to_tensor(self.data.iloc[idx, 1:-1].values.astype(np.uint8).reshape((1, 16, 16)))
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img.float(), label


class CustomImageDatasetUnlabeled(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.locs = glob.glob(self.img_dir + "/*")

    def __len__(self):
        return len(self.locs)

    def __getitem__(self, idx):
        img_path = self.locs[idx]
        img = Image.open(img_path)
        img = img.resize((32, 32), Image.ANTIALIAS)
        # image = torchvision.transforms.functional.to_tensor(self.data.iloc[idx, 1:-1].values.astype(np.uint8).reshape((1, 16, 16)))
        if self.transform:
            img = self.transform(img)
        return img.float()

full_dataset = CustomImageDataset(annotations_file=str(root /"dev"/"machine_learning"/"numberRecognition"/'green_digit_labels.csv'), img_dir='DigitsGreen', transform=ToTensor())
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

batch_size = 3

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')


def create_new_test_loader(dir):
    dataset = CustomImageDatasetUnlabeled(img_dir=dir, transform=ToTensor())
    return DataLoader(dataset, batch_size=3, shuffle=False, num_workers=0)
