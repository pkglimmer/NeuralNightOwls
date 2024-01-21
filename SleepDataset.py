import torch
import numpy as np
import glob
import os
from torch.utils.data import Dataset, DataLoader


class SleepSegmentationDataset(Dataset):
    def __init__(self, data_path, split='train'):
        data_files = glob.glob(os.path.join(data_path, '*_X.npy'))
        val_size = 5
        data_files = data_files[val_size:] if split == 'train' else data_files[:val_size]
        print(f'split: {split}, data_files: {len(data_files)}')
        label_files = [s.replace('_X.npy', '_y.npy') for s in data_files]
        # Load all data files
        self.data = [np.load(f) for f in data_files]  
        # Load all label files
        self.labels = [np.load(f)-1 for f in label_files] # Labels are 1-indexed originally

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        return x, self.labels[idx]
    
    
class SleepDataset(Dataset):
    def __init__(self, data_path, labels_path):
        self.data = np.load(data_path)
        self.labels = np.load(labels_path) - 1
        self.norm = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.data[idx]
        if self.norm:
            for i in range(x.shape[0]):
                x[i,:] = (x[i,:] - x[i,:].min()) / (x[i,:].max() - x[i,:].min() + 1e-6)

        return x, self.labels[idx]
    
    def get_labels(self):
        return self.labels


# Example usage
if __name__ == '__main__':
    data_base_path = '/rdf/user/pg34/sleep_data/processed'
    data_path = f'{data_base_path}/val_data.npy'
    labels_path = f'{data_base_path}/val_labels.npy'
    dataset = SleepDataset(data_path, labels_path)
    print(len(dataset))
    # train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

