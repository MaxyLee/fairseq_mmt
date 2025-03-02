# please see scripts/README.md firstly. 
import timm
import os
import torch
from tqdm import tqdm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import argparse
from PIL import Image

dic = {
    'test2017': 'test2017', 
    'testcoco': 'testcoco',
    'test2016': 'flickr30k',
    'train': 'flickr30k',
    'val': 'flickr30k',
    }

dic1 = {
    'test2017': 'test_2017_flickr.txt',
    'testcoco': 'test_2017_mscoco.txt',
    'test2016': 'test_2016_flickr.txt',
    'train': 'train.txt',
    'val': 'val.txt',
    }

dic2 = {
    'test2017': 'test1', 
    'testcoco': 'test2',
    'test2016': 'test',
    'train': 'train',
    'val': 'valid',
    }

dic_model = [
    'vit_tiny_patch16_384',
    'vit_small_patch16_384',
    'vit_base_patch16_384',
    'vit_large_patch16_384',
]

def get_filenames(path):
    l = []
    with open(path, 'r') as f:
        for line in f:
            fn = line.strip().split('#')[0]
            if fn not in l:
                l.append(fn)
    return l

if __name__ == "__main__":
    # please see scripts/README.md firstly. 
    # parser = argparse.ArgumentParser(description='which dataset')
    # parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test2016', 'test2017', 'testcoco'], help='which dataset')
    # parser.add_argument('--path', type=str)
    # parser.add_argument('--model', type=str)
    # args = parser.parse_args()

    # /path/flickr30k
    root_path = '/data/home/yc27434/projects/mmt/data/multi30k-dataset/data/task1'
    splits = ['train', 'val', 'test']
    image_path = f'{root_path}/flickr30k-images'
    filname_path = '/data/home/yc27434/projects/mmt/data/zero/Flickr30k-CNA'
    model_name = 'vit_base_patch16_384'
    
    model = timm.create_model(model_name, pretrained=True, num_classes=0).to('cuda:0') # if use cpu, uncomment '.to('cuda:0')'
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    save_dir = f'{root_path}/vit_feature/flickr30k-en2zh/{model_name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for split in splits:
        print('extracting ' + split + '\'s image feature from '+model_name) 
        
        tmp = []
        count = 1

        filenames = get_filenames(f'{filname_path}/images-{split}.txt')
        
        with torch.no_grad():
            for fn in tqdm(filenames):
                img = Image.open(f'{image_path}/{fn}').convert("RGB")
                input = transform(img).unsqueeze(0).to('cuda:0') # transform and add batch dimension
                
                out = model.forward_features(input)
                
                tmp.append(out.detach().cpu())
                if len(tmp) == 2000:
                    res = torch.cat(tmp)
                    print(res.shape)
                    torch.save(res, f'{save_dir}/{count}{split}.pth')
                    count += 1
                    tmp = []
        
        res = torch.cat(tmp)
        if count > 1:
            torch.save(res, f'{save_dir}/final{split}.pth')
        else:
            print('feature shape:', res.shape, ',save in:', save_dir+'/'+split+'.pth')
            torch.save(res, f'{save_dir}/{split}.pth')
        del tmp
        
        res = None
        if count > 1:
            for i in range(1, count):
                
                _tmp = torch.load(f'{save_dir}/{i}{split}.pth')
                if res is None:
                    res = _tmp
                else:
                    res = torch.cat([res, _tmp])
                    del _tmp
            
            print('feature shape:', res.shape, ',save in:', save_dir+'/'+split+'.pth')
            torch.save(res, f'{save_dir}/{split}.pth')
            
            # delete  
            for i in range(1, count):
                os.remove(f'{save_dir}/{i}{split}.pth')
            os.remove(f'{save_dir}/final{split}.pth')
        
# if __name__ == "__main__":
#     # please see scripts/README.md firstly. 
#     parser = argparse.ArgumentParser(description='which dataset')
#     parser.add_argument('--dataset', type=str, choices=['train', 'val', 'test2016', 'test2017', 'testcoco'], help='which dataset')
#     parser.add_argument('--path', type=str)
#     parser.add_argument('--model', type=str)
#     args = parser.parse_args()

#     # /path/flickr30k
#     flickr30k_path = args.path
#     dataset = args.dataset
#     model_name = args.model
#     save_dir = os.path.join('data', model_name)
    
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     print('extracting ' + dataset + '\'s image feature from '+model_name) 
#     model = timm.create_model(model_name, pretrained=True, num_classes=0).to('cuda:0') # if use cpu, uncomment '.to('cuda:0')'
#     model.eval()
#     config = resolve_data_config({}, model=model)
#     transform = create_transform(**config)

#     tmp = []
#     count = 1

#     filenames = get_filenames(os.path.join(flickr30k_path, dic1[dataset]))
    
#     with torch.no_grad():
#         for i in tqdm(filenames):
#             i = os.path.join(flickr30k_path, dic[dataset]+'-images', i)
#             img = Image.open(i).convert("RGB")
#             input = transform(img).unsqueeze(0).to('cuda:0') # transform and add batch dimension
            
#             out = model.forward_features(input)
            
#             tmp.append(out.detach().to('cuda:0'))
#             if len(tmp) == 2000:
#                 res = torch.cat(tmp).cpu()
#                 print(res.shape)
#                 torch.save(res, os.path.join(save_dir, str(count)+dic2[dataset]+'.pth'))
#                 count += 1
#                 tmp = []
    
#     res = torch.cat(tmp).cpu()
#     if count > 1:
#         torch.save(res, os.path.join(save_dir, 'final'+dic2[dataset]+'.pth'))
#     else:
#         print('feature shape:', res.shape, ',save in:', save_dir+'/'+dic2[dataset]+'.pth')
#         torch.save(res, os.path.join(save_dir, dic2[dataset]+'.pth'))
#     del tmp
    
#     _tmp = []
#     if count > 1:
#         for i in range(1, count):
#             _tmp.append(torch.load(os.path.join(save_dir, str(i)+dic2[dataset]+'.pth')))
#         _tmp.append(torch.load(os.path.join(save_dir, 'final'+dic2[dataset]+'.pth')))
#         res = torch.cat(_tmp).cpu()
#         print('feature shape:', res.shape, ',save in:', save_dir+'/'+dic2[dataset]+'.pth')
#         torch.save(res, os.path.join(save_dir, dic2[dataset]+'.pth'))
        
#         # delete  
#         for i in range(1, count):
#             os.remove(os.path.join(save_dir, str(i)+dic2[dataset]+'.pth'))
#         os.remove(os.path.join(save_dir, 'final'+dic2[dataset]+'.pth'))