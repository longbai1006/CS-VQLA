import os
import sys
import h5py
import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from glob import glob
import torch
from torch import nn
from torchvision import models
from tqdm import tqdm

class FeatureExtractor(nn.Module):
    def __init__(self, patch_size = 4):
        super(FeatureExtractor, self).__init__()
        # visual feature extraction
        self.img_feature_extractor = models.resnet18(pretrained=True)
        self.img_feature_extractor = torch.nn.Sequential(*(list(self.img_feature_extractor.children())[:-2]))
        self.resize_dim = nn.AdaptiveAvgPool2d((patch_size,patch_size))
        
    def forward(self, img):
        outputs = self.resize_dim(self.img_feature_extractor(img))
        return outputs

# input data and IO folder location
filenames = []
# seq = ['1','2','3','4','5','6','7','9','10','11','12','14','15','16']
# folder_head = '/home/ren2/data3/Long/Surgical_VQLA/dataset/EndoVis-18-VQA/'
# folder_tail = '/*.png'

seq = ['train'] # , 'test']
folder_head = '/home/ren2/data3/Long/Surgical_VQA_Detection/Continual_VQA/dataset/m2cai2016/'
folder_tail = '/*.jpg'

for curr_seq in seq: 
    # filenames = filenames + glob(folder_head + 'seq_' + str(curr_seq) + '/left_frames' + folder_tail)
    filenames = filenames + glob(folder_head + str(curr_seq) + '/img' + folder_tail)
    print(folder_head + str(curr_seq) + '/img' + folder_tail)
new_filenames = []
for filename in filenames:
    # frame_num = int(filename.split('/')[-1].split('.')[0].strip('frame'))
    # if frame_num % 1 == 0: new_filenames.append(filename)
    new_filenames.append(filename)

transform = transforms.Compose([transforms.Resize((300,256)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                                ])

# arguments
parser = argparse.ArgumentParser(description='feature extractor')
parser.add_argument('--patch_size',        type=int,       default=5,           help='')
args = parser.parse_args()

    
# declare fearure extraction model
feature_network = FeatureExtractor(patch_size = args.patch_size)

# Set data parallel based on GPU
num_gpu = torch.cuda.device_count()
if num_gpu > 0:
    device_ids = np.arange(num_gpu).tolist()
    feature_network = nn.DataParallel(feature_network, device_ids=device_ids)

# Use Cuda
feature_network = feature_network.cuda()
feature_network.eval()


for img_loc in tqdm(new_filenames):
    
    # get visual features
    img = Image.open(img_loc)
    img = transform(img)
    img = torch.unsqueeze(img, 0)
    with torch.no_grad():
        visual_features = feature_network(img)
        visual_features = torch.flatten(visual_features, start_dim=2)
        visual_features = visual_features.permute((0,2,1))   
        visual_features = visual_features.squeeze(0)
        visual_features = visual_features.data.cpu().numpy()

    # save extracted features
    img_loc = img_loc.split('/')
    save_dir = '/' + os.path.join(img_loc[0],img_loc[1],img_loc[2],img_loc[3],img_loc[4],img_loc[5],img_loc[6],img_loc[7],img_loc[8],img_loc[9],'vqla/img_features',(str(args.patch_size)+'x'+str(args.patch_size)))
    if not os.path.exists(save_dir):os.makedirs(save_dir)
    
    # save to file
    hdf5_file = h5py.File(os.path.join(save_dir, '{}.hdf5'.format(img_loc[-1].split('.')[0])),'w')
    print(os.path.join(save_dir, '{}.hdf5'.format(img_loc[-1].split('.')[0])))
    hdf5_file.create_dataset('visual_features', data=visual_features)
    hdf5_file.close()
    print('save_dir: ', save_dir, ' | visual_features: ', visual_features.shape)
