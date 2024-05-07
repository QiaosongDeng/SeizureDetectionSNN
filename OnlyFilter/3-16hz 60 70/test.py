import mne
import os
import pandas as pd
import numpy as np

import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms

from sklearn.model_selection import train_test_split

# dataloader arguments
batch_size = 128
# file_path = './chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_01.edf'
file_path = './Dataset'
targets_path = './labels.csv'

dtype = torch.float
print(torch.cuda.is_available())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def read_edf(file_path):
    """Read an EDF file and return the data and times."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    data, times = raw[:, :]
    return data, times

def find_edf_files(directory):
    """Find all .edf files in the given directory."""
    edf_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".edf"):
                edf_files.append(os.path.join(root, file))
    return edf_files

def combine_edf_data(directory):
    """Combine data from all EDF files in a directory."""
    edf_files = find_edf_files(directory)
    combined_data = []
    combined_times = []

    for file in edf_files:
        data, times = read_edf(file)
        combined_data.append(data)
        combined_times.append(times)

    #concatenate along the time axis (second dimension)
    combined_data = np.concatenate(combined_data, axis=1)
    combined_times = np.concatenate(combined_times)

    return combined_data, combined_times


data, times = combine_edf_data(file_path)
data = data.T # Transpose if each sample should be (timepoints, channels)
print("data type: " + str(type(data)) + " data size: " + str(data.shape))
print("times type: " + str(type(data)) + " times size: " + str(times.shape))

# Read the targets CSV file, note this CSV file should have no header
df = pd.read_csv(targets_path, header=None)
targets = df.values
print("targets type: " + str(type(targets)) + " targets size: " + str(targets.shape))
print("targets type: " + str(type(targets[0,0])) + " targets size: " + str(targets.shape))

data = data.reshape(targets.shape[0], 256, 23) #(second, timepoints, channels)
print("reshaped_data type: " + str(type(data)) + " reshaped_data size: " + str(data.shape))

# Extract EEG data where target is 1
data_target_1 = data[targets[:, 0] == 1]

# Extract EEG data where target is 0 and randomly select same number of rows as data_target_1
data_target_0 = data[targets[:, 0] == 0]
np.random.shuffle(data_target_0)
data_target_0 = data_target_0[:len(data_target_1)]

# Combine the selected data
selected_data = np.vstack((data_target_1, data_target_0))

# Create corresponding targets for the selected data
selected_targets = np.array([1] * len(data_target_1) + [0] * len(data_target_0), dtype=np.int64).reshape(-1, 1)

print("selected data type: " + str(type(selected_data)) + " selected data size: " + str(selected_data.shape))
print("selected targets type: " + str(type(selected_targets)) + " selected targets size: " + str(selected_targets.shape))

del data, times, targets

from scipy.signal import butter, filtfilt

# 定义带通滤波器函数
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # Nyquist 频率
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# 应用带通滤波器
def apply_bandpass(data, lowcut=3, highcut=16, fs=256, order=5):
    filtered_data = np.zeros_like(data)
    num_samples, _, num_channels = data.shape

    for i in range(num_samples):
        for j in range(num_channels):
            filtered_data[i, :, j] = bandpass_filter(data[i, :, j], lowcut, highcut, fs, order)

    return filtered_data

selected_data = apply_bandpass(selected_data)
print("filtered_data type: " + str(type(selected_data)) + " filtered_data size: " + str(selected_data.shape))


class EEGNormalize(object):
    def __call__(self, sample):
        # sample shape: (256, 23)
        normalized_sample = np.zeros_like(sample)
        # Iterate over channels
        for i in range(sample.shape[1]):  # sample.shape[1] is 23
            channel_data = sample[:, i]
            channel_mean = channel_data.mean()
            channel_std = channel_data.std()
            normalized_sample[:, i] = (channel_data - channel_mean) / channel_std
        return normalized_sample

# Define transformations
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    EEGNormalize(),
    # Add other transformations as needed
])

class EDFDataset(Dataset):
    def __init__(self, data, targets, transforms=None):
        self.data = data
        self.transforms = transforms
        self.targets = targets

        assert len(self.data) == len(self.targets), "Data and targets must be the same length"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        # print(f"Before transform: {sample.shape}")  # Before transform

        # Reshape the sample to 2D if it's not already
        if sample.ndim != 2:
            raise ValueError("Sample must be 2D")

        if self.transforms:
            sample = self.transforms(sample)
        # print(f"After transform: {sample.shape}")  # After transform
        
        return sample.squeeze(), self.targets[index].squeeze()

# Create test dataset
test_dataset  = EDFDataset(selected_data, selected_targets, transforms=data_transforms)

# Create DataLoader
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

# Network Architecture
num_inputs = 23
num_hidden = 50
num_outputs = 2 

# Temporal Dynamics
num_steps = 256
beta = 0.95

# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # 添加这行来检查输入形状
        # x should be of shape (batch_size, 256, 23)
        
        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[:, step, :])  # Process each time step
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        # Stack the recorded spikes and membrane potentials over the time steps
        spk2_rec = torch.stack(spk2_rec, dim=1)
        # print(f"spk2_rec shape: {spk2_rec.shape}")
        mem2_rec = torch.stack(mem2_rec, dim=1)
        # print(f"mem2_rec shape: {mem2_rec.shape}")

        return spk2_rec, mem2_rec

        
# Load the network onto CUDA if available
net = Net().to(device)

net.load_state_dict(torch.load('trained_model.pth'))


# 初始化计数器
TP = 0  # 真阳性（True Positives）
FP = 0  # 假阳性（False Positives）
FN = 0  # 假阴性（False Negatives）
TN = 0  # 真阴性（True Negatives）


with torch.no_grad():
    net.eval()
    for data, targets in test_loader:
        data = data.to(device).float()
        targets = targets.to(device)

        # 前向传播
        spk_rec, _ = net(data)

        # 将所有时间步的尖峰相加，获取预测
        # spk_rec 形状为 (batch_size, num_steps, num_outputs)
        # 对时间步（dim=1）进行求和
        total_spikes = spk_rec.sum(dim=1)
        _, predicted = total_spikes.max(1)

        # 更新计数器
        TP += ((predicted == 1) & (targets == 1)).sum().item()
        FP += ((predicted == 1) & (targets == 0)).sum().item()
        FN += ((predicted == 0) & (targets == 1)).sum().item()
        TN += ((predicted == 0) & (targets == 0)).sum().item()

# 计算灵敏度（sensitivity）和特异性（specificity）
sensitivity = TP / (TP + FN) if TP + FN > 0 else 0
specificity = TN / (TN + FP) if TN + FP > 0 else 0

# 计算整体精度（accuracy）
total = TP + TN + FP + FN
correct = TP + TN
accuracy = correct / total if total > 0 else 0

print(f"Accuracy: {accuracy*100:.2f}%")
print(f"Sensitivity: {sensitivity*100:.2f}%")
print(f"Specificity: {specificity*100:.2f}%")
print(f"Total correctly classified test set samples: {correct}/{total}")

