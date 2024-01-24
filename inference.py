import torch
import torch.nn as nn
import numpy as np
import os
from scipy import stats

from models.unet import UNet1D
from models.resnet1d import resnet18, resnet10, resnext50_32x4d, resnet34, resnet50
from utils.ts_reshape import one_hot_encode, repeat_labels


os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# IMPORTANT: Output labels need to be added 1 before saving because the labels are 0-indexed

def load_wandb_state_dict(ckpt_path):
    # Load the state dictionary from the checkpoint
    state_dict = torch.load(ckpt_path)['state_dict']
    # Create a new state dictionary with the "model." prefix removed from the keys
    new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    return new_state_dict


def resnet10_inference(input, ckpt_path=None):
    # input: (B, 6, 3000)
    # output: (B, 6, 3000)
    ckpt_path = os.path.join('/rdf/user/pg34/NeuroTech/ckpt/R10', ckpt_path)
    model = resnet10(num_input_channels=6, num_classes=6)
    model.load_state_dict(load_wandb_state_dict(ckpt_path))
    # min-max normalize input    
    for i in range(input.shape[0]):  # Loop over the batch dimension
        for j in range(input.shape[1]):  # Loop over the channel dimension
            input[i, j, :] = (input[i, j, :] - input[i, j, :].min()) / (input[i, j, :].max() - input[i, j, :].min() + 1e-6)
    
    outputs = model(input)
    _, preds = torch.max(outputs, dim=1)
    print(f'ResNet10 inference complete. Output shape: {preds.shape}')
    return preds

def resnet18_inference(input, ckpt_path=None):
    # input: (B, 6, 3000)
    # output: (B, 6, 3000)
    ckpt_path = os.path.join('/rdf/user/pg34/NeuroTech/ckpt/R18', ckpt_path)
    model = resnet18(num_input_channels=6, num_classes=6)
    model.load_state_dict(load_wandb_state_dict(ckpt_path))
    # min-max normalize input    
    for i in range(input.shape[0]):  # Loop over the batch dimension
        for j in range(input.shape[1]):  # Loop over the channel dimension
            input[i, j, :] = (input[i, j, :] - input[i, j, :].min()) / (input[i, j, :].max() - input[i, j, :].min() + 1e-6)
    
    outputs = model(input)
    _, preds = torch.max(outputs, dim=1)
    print(f'ResNet18 inference complete. Output shape: {preds.shape}')
    return preds

def resnet50_inference(input, ckpt_path=None):
    # input: (B, 6, 3000)
    # output: (B, 6, 3000)
    ckpt_path = os.path.join('/rdf/user/pg34/NeuroTech/ckpt/R50', ckpt_path)
    model = resnet50(num_input_channels=6, num_classes=6)
    model.load_state_dict(load_wandb_state_dict(ckpt_path))
    # min-max normalize input    
    for i in range(input.shape[0]):  # Loop over the batch dimension
        for j in range(input.shape[1]):  # Loop over the channel dimension
            input[i, j, :] = (input[i, j, :] - input[i, j, :].min()) / (input[i, j, :].max() - input[i, j, :].min() + 1e-6)
    
    outputs = model(input)
    _, preds = torch.max(outputs, dim=1)
    print(f'ResNet50 inference complete. Output shape: {preds.shape}')
    return preds

def unet_inference(input, ckpt_path=None, downsample_ratio=2):
    ckpt_path = os.path.join('/rdf/user/pg34/NeuroTech/ckpt/unet_v0', ckpt_path)
    model = UNet1D()
    model.load_state_dict(load_wandb_state_dict(ckpt_path))
    
    x = input.unsqueeze(0)
    batch_size = x.size(0)
    downsample_ratio = 2
    interval_len = 3000 // downsample_ratio
    x = x.permute(0, 2, 1, 3).reshape(batch_size, 6, -1)[:,:,::downsample_ratio]
    mean = x.mean(dim=-1, keepdim=True)
    std = x.std(dim=-1, keepdim=True)
    x = (x - mean) / (std + 1e-6)
    outputs = model(x)
    # print(outputs.shape, y_mask.shape) # torch.Size([1, 6, 84120]) torch.Size([1, 84120, 6])
    outputs = outputs.permute(0, 2, 1)
    
    # Step 1: Argmax across channels to get most likely class
    predicted_classes = torch.argmax(outputs, dim=2)  # outputs Shape (B, seq_len, 6)

    # Step 2: Majority vote (mode) for every 30 indices
    B, seq_len = predicted_classes.shape
    reduced_seq_len = seq_len // interval_len
    preds = torch.zeros((B, reduced_seq_len), dtype=predicted_classes.dtype)
    # print(preds.shape, y.shape, end='\n\n\n\n\n')
    for i in range(reduced_seq_len):
        sliced = predicted_classes[:, i*interval_len:(i+1)*interval_len]
        mode_values, _ = stats.mode(sliced.cpu(), axis=1)
        preds[:, i] = torch.from_numpy(mode_values).squeeze()
    print(f'UNet inference complete. Output shape: {preds.squeeze(0).shape}')
    return preds.squeeze(0)
    
    
def test_inference():
    x = torch.zeros(1300, 6, 3000)
    unet_inference(x).shape
    resnet10_inference(x).shape
    resnet50_inference(x).shape

    
def inference(x):
    unet_ckpt_paths = os.listdir('/rdf/user/pg34/NeuroTech/ckpt/unet_v0')
    y1 = unet_inference(x, ckpt_path=unet_ckpt_paths[0], downsample_ratio=5)
    y2 = unet_inference(x, ckpt_path=unet_ckpt_paths[1], downsample_ratio=10)
    
    resnet10_ckpt_paths = os.listdir('/rdf/user/pg34/NeuroTech/ckpt/R10')
    y3 = resnet10_inference(x, ckpt_path=resnet10_ckpt_paths[0])
    y4 = resnet10_inference(x, ckpt_path=resnet10_ckpt_paths[1])
    
    resnet18_ckpt_paths = os.listdir('/rdf/user/pg34/NeuroTech/ckpt/R18')
    y5 = resnet18_inference(x, ckpt_path=resnet18_ckpt_paths[0])
    y6 = resnet18_inference(x, ckpt_path=resnet18_ckpt_paths[1])
    
    resnet50_ckpt_paths = os.listdir('/rdf/user/pg34/NeuroTech/ckpt/R50')
    y7 = resnet50_inference(x, ckpt_path=resnet50_ckpt_paths[0])
    
        # Stack the tensors
    stacked_tensors = torch.stack([y1, y2, y3, y4, y5, y6, y7], dim=0)  # This creates a tensor of shape [number_of_tensors, 1300]
    # stacked_tensors = torch.stack([y3, y4, y5, y6, y7], dim=0)  # ResNet ensemble
    # stacked_tensors = torch.stack([y1, y2], dim=0)  # UNet ensemble
    

    # Compute the mode along the first dimension
    mode_values, mode_indices = torch.mode(stacked_tensors, dim=0)
    return mode_values


if __name__ == '__main__':
    suffix = '_unet'
    test_data_path = '/rdf/user/pg34/sleep_data/Eval_new'
    x1 = np.load(os.path.join(test_data_path, 'eval_a_NEW_X.npy'))
    y1 = inference(torch.from_numpy(x1).float())
    y1 = y1 + 1
    np.save(f'assets/a_pred{suffix}.npy', y1.numpy())
    print(y1)
    print(sum(y1))

    x2 = np.load(os.path.join(test_data_path, 'eval_b_NEW_X.npy'))
    y2 = inference(torch.from_numpy(x2).float())
    y2 = y2 + 1
    np.save(f'assets/b_pred{suffix}.npy', y2.numpy())
    print(y2)
    print(sum(y2))

