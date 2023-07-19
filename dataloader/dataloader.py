import os
import glob
import h5py
import cv2
import torch
from torch.utils.data import Dataset
from utils.vqla import *

'''
EndoVis18 VQLA dataloader
'''
class EndoVis18VQLA(Dataset):
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqla/label/*.txt'
    	patch_size      = 5
    '''
    def __init__(self, seq, folder_head, folder_tail, patch_size=4):
             
        self.patch_size = patch_size
        
        # files, question and answers
        filenames = []
        for curr_seq in seq: filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('EndoVis18 Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom', 
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 'clip_applier',
                        'suction', 'stapler',  # EndoVis18 # base 26
                        'grasping_retractor', 'vessel_sealer', # EndoVis17 # step1 2
                        'irrigator', 'hook', 'grasper', 'specimenbag', 'clipper'] # Cholec # step2 5

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = '/' + os.path.join(loc[0],loc[1],loc[2],loc[3],
         'vqla/img_features', (str(self.patch_size)+'x'+str(self.patch_size)),loc[-1].split('_')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
        # detection bbox
        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
        bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([1280, 1024, 1280, 1024])
        return loc[-4] + '/' + loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize

class EndoVis17VQLA(Dataset):

    def __init__(self, external_folder_head, external_folder_tail, patch_size=4):
             
        self.patch_size = patch_size
        
        # files, question and answers
        filenames = []
        filenames =  glob.glob(external_folder_head + external_folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('EndoVis17 Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom', 
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 'clip_applier',
                        'suction', 'stapler',  # EndoVis18 # base 26
                        'grasping_retractor', 'vessel_sealer', # EndoVis17 # step1 2
                        'irrigator', 'hook', 'grasper', 'specimenbag', 'clipper'] # Cholec # step2 5

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = '/' + os.path.join(loc[0],loc[1],loc[2],loc[3],
        'vqla/img_features', (str(self.patch_size)+'x'+str(self.patch_size)),loc[-1].split('.')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
            
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))
        
        # detection bbox
        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
        bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([1280, 1024, 1280, 1024])
        return loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize

class M2CAIVQLA(Dataset):

    def __init__(self, external_folder_head, external_folder_tail, patch_size=4):
             
        self.patch_size = patch_size
        
        # files, question and answers
        filenames = []
        filenames =  glob.glob(external_folder_head + external_folder_tail)
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines: self.vqas.append([file, line])
        print('M2CAI Total files: %d | Total question: %.d' %(len(filenames), len(self.vqas)))
        
        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                        'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction', 
                        'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                        'left-top', 'right-top', 'left-bottom', 'right-bottom', 
                        'bipolar_forceps', 'monopolar_curved_scissors', 'prograsp_forceps', 
                        'large_needle_driver', 'ultrasound_probe', 'clip_applier',
                        'suction', 'stapler',  # EndoVis18 # base 26
                        'grasping_retractor', 'vessel_sealer', # EndoVis17 # step1 2
                        'irrigator', 'hook', 'grasper', 'specimenbag', 'clipper'] # Cholec # step2 5


    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        
        # img
        loc = self.vqas[idx][0].split('/')
        visual_feature_loc = '/' + os.path.join(loc[0],loc[1],loc[2],loc[3],
        'vqla/img_features', (str(self.patch_size)+'x'+str(self.patch_size)),loc[-1].split('.')[0]+'.hdf5')
        frame_data = h5py.File(visual_feature_loc, 'r')    
        visual_features = torch.from_numpy(frame_data['visual_features'][:])
        
        img_loc = '/' + os.path.join(loc[0],loc[1],loc[2],loc[3],
        'img',loc[-1].split('.')[0]+'.jpg')
        img = cv2.imread(img_loc)
        img_height, img_width, _ = img.shape 
        
        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        # detection bbox
        bbox_label = self.vqas[idx][1].split('|')[2].replace(' ','').split(',')
        bbox_label_xyxy = torch.tensor(list(map(int, bbox_label))) - torch.tensor([1, 1, 1, 1])
        bbox_label_cxcywh = box_xyxy_to_cxcywh(bbox_label_xyxy)
        bbox_label_cxcywh_nomalize = bbox_label_cxcywh / torch.tensor([img_width, img_height, img_width, img_height])
        return loc[-1].split('_')[0], visual_features, question, label, bbox_label_cxcywh_nomalize
