import numpy as np
from sklearn.model_selection import train_test_split
import os
import glob
from tqdm import tqdm

def preprocess_and_save(data_dir, save_dir):
    all_train_data = []
    all_train_labels = []
    all_val_data = []
    all_val_labels = []

    # Loop through each pair of files
    for x_file in tqdm(glob.glob(os.path.join(data_dir, '*_X.npy'))):
        y_file = x_file.replace('_X.npy', '_y.npy')

        # Load data and labels
        data = np.load(x_file)
        labels = np.load(y_file).astype(int)

        # Split data (stratified if possible)
        # Split data (stratify if possible)
        if np.min(np.bincount(labels)) < 3:
            stratify = None
        else:
            stratify = labels
        train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.1, stratify=stratify)

        # Append to lists
        all_train_data.append(train_data)
        all_train_labels.append(train_labels)
        all_val_data.append(val_data)
        all_val_labels.append(val_labels)

    # Merge and save data
    np.save(os.path.join(save_dir, 'train_data.npy'), np.concatenate(all_train_data, axis=0))
    np.save(os.path.join(save_dir, 'train_labels.npy'), np.concatenate(all_train_labels, axis=0))
    np.save(os.path.join(save_dir, 'val_data.npy'), np.concatenate(all_val_data, axis=0))
    np.save(os.path.join(save_dir, 'val_labels.npy'), np.concatenate(all_val_labels, axis=0))

# Usage example
data_dir = '/rdf/user/pg34/sleep_data/Training_new'
save_dir = '/rdf/user/pg34/sleep_data/processed_new'
preprocess_and_save(data_dir, save_dir)
