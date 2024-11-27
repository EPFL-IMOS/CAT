import os
import torch
from scipy.io import loadmat
from scipy.interpolate import interp1d
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

signal_size = 1024


def res(array, npts):
    interpolated = interp1d(np.arange(len(array)), array, axis = 0, fill_value = 'extrapolate')
    downsampled = interpolated(np.linspace(0, len(array), npts))
    return downsampled


class ISU(torch.utils.data.Dataset):
    classes = ["combine", "inner", "none", "outer"]
    num_classes = len(classes)

    def __init__(self, root: str, train=True, transform=None):

        self.transform = transform
        
        if train:
            path = os.path.join(root, "Generated_data", "train", "Raw")
        else:
            path = os.path.join(root, "Generated_data", "test", "Raw")

        self.samples = []
        self.labels = []
        for i, c in tqdm(enumerate(self.classes)):
            samples, labels = self.data_load(os.path.join(path, f"{c}_defect12800.mat"), f"data{c}_defect", i)
            self.samples += samples
            self.labels += labels

    def data_load(self, filename, name, label):
        '''
        This function is mainly used to generate test data and training data.
        filename:Data location
        '''
        fl = loadmat(filename)[name]
        fl = fl[:,:-2]  #Take out the data
        data = [] 
        lab = []
        for i in range(fl.shape[0]):
            start,end=0,signal_size
            while end <= fl.shape[1]:
                window = fl[i, start:end]
                window = np.array(window)
                window = window.reshape(-1, 1)
                data.append(window)
                lab.append(label)
                start += signal_size
                end += signal_size

        return data, lab

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
            return (tuple): (x, target) where target is index of the target class.
        """
        x, target = self.samples[index], self.labels[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, target

    def __len__(self) -> int:
        return len(self.samples)


class ISU2(torch.utils.data.Dataset):
    classes = ["Combine", "Inner", "Healthy", "Outer"]
    tasks = {
        "1": "Load0/35Hz",
        "2": "Load1/35Hz",
        "3": "Load0/25Hz",
        "4": "Load1/25Hz",
    }
    num_classes = len(classes)

    def __init__(self, root: str, task: str, train=True, transform=None):

        self.transform = transform
        
        path = os.path.join(root, self.tasks[task])

        self.samples = []
        self.labels = []
        for i, c in tqdm(enumerate(self.classes)):
            samples, labels = self.data_load(os.path.join(path, c), i)
            self.samples += samples
            self.labels += labels
        
        # Train/val split (fixed 80/20 train/val split)
        random.seed(42)
        random.shuffle(self.samples)
        random.seed(42)
        random.shuffle(self.labels)
        split = int(0.8 * len(self.samples))
        if train:
            self.samples = self.samples[:split]
            self.labels = self.labels[:split]
        else:
            self.samples = self.samples[split:]
            self.labels = self.labels[split:]

    def data_load(self, path, label):
        '''
        This function is mainly used to generate test data and training data.
        filename:Data location
        '''
        data = [] 
        lab = []
        for f in os.listdir(path):
            start,end=0,signal_size
            # x = xlrd.open_workbook(os.path.join(path, f))
            # x = x.sheet_by_name('sheet1')
            # x = x.col_values(0)[1:]
            # x = pd.read_parquet(os.path.join(path, f))["Voltage"].values
            x = np.load(os.path.join(path, f))
            # resample 25.6kHz -> 12.8 kHz
            x = res(x, len(x)//2)
            while end <= len(x):
                window = x[start:end]
                window = window.reshape(-1, 1)
                data.append(window)
                lab.append(label)
                start += signal_size
                end += signal_size

        return data, lab

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index
            return (tuple): (x, target) where target is index of the target class.
        """
        x, target = self.samples[index], self.labels[index]
        if self.transform is not None:
            x = self.transform(x)
        return x, target

    def __len__(self) -> int:
        return len(self.samples)
