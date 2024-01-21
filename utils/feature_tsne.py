import os
import argparse
from time import time
import torch
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sys
from torch.utils.data import DataLoader
sys.path.append("/home/pg34/code/seizure/selective_sampling")
from models.resnet1d import *
from MSGDataset import MsgRawData, BCMDataset

class T_sne_visual():
    def __init__(self, model, dataset, dataloader, fig_path, fig_title):
        self.model = model
        self.dataset = dataset
        self.dataloader = dataloader
        self.save_path = fig_path
        self.fig_title = fig_title
        # self.class_list = dataset.classes
    def visual_dataset(self):
        imgs = []
        labels = []
        for img, label in self.dataset:
            imgs.append(np.array(img).transpose((2, 1, 0)).reshape(-1))
            # tag = self.class_list[label]
            labels.append(label)
        self.t_sne(np.array(imgs), labels,title=f'Dataset visualize result\n')

    def visual_feature_map(self, layer):
        self.model.eval()
        with torch.no_grad():
            self.feature_map_list = []
            labels = []
            getattr(self.model, layer).register_forward_hook(self.forward_hook)
            for x, label in self.dataloader:
                x = x.to('cuda:0', dtype=torch.float32)
                self.model(x)
                labels += label.tolist()
            self.feature_map_list = torch.cat(self.feature_map_list, dim=0)
            self.feature_map_list = torch.flatten(self.feature_map_list, start_dim=1)
            self.t_sne(np.array(self.feature_map_list.cpu()), np.array(labels), title=f'{self.fig_title} feature tSNE\n', save_path=self.save_path)

    def forward_hook(self, model, input, output):
        self.feature_map_list.append(output)

    def set_plt(self, start_time, end_time,title):
        plt.title(f'{title} time consume:{end_time - start_time:.3f} s')
        plt.legend(title='')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks([])
        plt.yticks([])

    def t_sne(self, data, label, title, save_path):
        print('starting T-SNE process')
        start_time = time()
        data = TSNE(n_components=2, learning_rate='auto', init='pca').fit_transform(data)
        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        df = pd.DataFrame(data, columns=['x', 'y']) 
        df.insert(loc=1, column='label', value=label)
        end_time = time()
        print('Finished')

        # plot
        sns.scatterplot(x='x', y='y', hue='label', s=3, palette="Set2", data=df)
        self.set_plt(start_time, end_time, title)
        plt.savefig(save_path, dpi=400)
        plt.show()
    
def main():
    patient_id = 4
    # dataset = MsgRawData(patient_id, mode='val', split='val', norm=True, undersample=True)
    dataset = MsgRawData(patient_id, mode='val', split='all', norm=True, undersample=True)
    val_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=64, persistent_workers=True)
    model = resnet18(num_input_channels=5, num_classes=2).cuda()
    ckpt_path = f'/home/pg34/code/seizure/selective_sampling/ckpt/model_recon_dagmm_best_p{patient_id}.pt'
    recon_state_dict = torch.load(ckpt_path, map_location='cuda')
    new_state_dict = {}
    for key, value in recon_state_dict['encoder'].items():
        if key.startswith('encoder'):
            new_state_dict[key[8:]] = value
    model.load_state_dict(new_state_dict)

    save_path = '/home/pg34/code/seizure/selective_sampling/assets/features'
    title = f'tsne_patient{patient_id}'
    save_path = os.path.join(save_path, title) + '.png'
    print(f'save path {save_path}')
    t = T_sne_visual(model, dataset, val_loader, save_path, title)
    # t.visual_feature_map('layer4')
    t.visual_feature_map('avgpool')
    
    
if __name__ == "__main__":
    main()
