import os
import torch

from glob import glob

class ImageDataset(torch.utils.data.Dataset):
    """
    For loading image datasets
    """
    def __init__(self, feat_path: str, mask_path: str, split_image_feat: bool):
        self.split_image_feat = split_image_feat
        
        if split_image_feat:
            
            self.img_feat_paths = glob(f'{feat_path}/*train.pth')
            self.split_num = 0
            self.img_feat = torch.load(self.img_feat_paths[0])
            self.split_size = self.img_feat.shape[0]
            self.img_feat_mask = None
            
            self.size = self.split_size * (len(self.img_feat_paths) - 1) + torch.load(self.img_feat_paths[-1]).shape[0]
        else:
            self.img_feat = torch.load(feat_path)
            self.img_feat_mask = None
            if os.path.exists(mask_path):
                self.img_feat_mask = torch.load(mask_path)

            self.size = self.img_feat.shape[0]

    def __getitem__(self, idx):
        if self.img_feat_mask is None:
            return self.get_img_feat(idx), None
        else:
            return self.get_img_feat(idx), self.img_feat_mask[idx]
        
    def get_img_feat(self, idx):
        if self.split_image_feat:
            num = int(idx / self.split_size)
            i = idx % self.split_size
            
            if num != self.split_num:
                self.split_num = num
                del self.img_feat
                self.img_feat = torch.load(self.img_feat_paths[num])
                
            return self.img_feat[i]
            
        else:
            return self.img_feat[idx]

    def __len__(self):
        return self.size
