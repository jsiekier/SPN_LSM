import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from scipy.ndimage import zoom

from utils import loadvideo
import pickle as pkl
class CachedVideoLoader():
    def __init__(self, path, deactivate=False):
        self.path = path
        self.video = {}
        self.deactivate = deactivate

    def __call__(self, fname):
        if self.deactivate:
            return loadvideo(os.path.join(self.path, fname)).astype(np.uint8) # type: ignore
        if not fname in self.video:
            self.video[fname] = loadvideo(os.path.join(self.path, fname)).astype(np.uint8) # type: ignore
        return self.video[fname]

class EchoVideo(Dataset):
    def __init__(self, config, splits=["TRAIN", "VAL", "TEST"], chunks=1, chunk=0):

        # Arguments:
        self.config = config
        self.splits = splits if isinstance(splits, list) else [splits]
        
        # Data paths
        self.data_path = config.dataset.data_path
        self.video_folder_path = os.path.join(config.dataset.data_path, "Videos")

        # Transformations
        self.target_fps = config.dataset.get("fps", 32)
        self.duration = config.dataset.get("duration", 2.0)
        self.videos_length = int(self.target_fps * self.duration)

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x/255.0)),
            transforms.Resize(max(config.imagen.image_sizes)),
        ])

        if config.dataset.get("grayscale", False):
            self.transform.transforms.extend([
                transforms.Lambda(lambda x: x.permute(1,0,2,3)), # 3 64 112 112 => 64 3 112 112
                transforms.Grayscale(num_output_channels=3),# type: ignore
                transforms.Lambda(lambda x: x.permute(1,0,2,3)), # 64 3 112 112 => 64 3 112 112
                ]) 

        # Load data points
        self.filelist = pd.read_csv(os.path.join(config.dataset.data_path, "FileList.csv"))
        new_fl = self.filelist[self.filelist['Split'] == ""]
        for split in self.splits:
            new_fl = pd.concat((new_fl, self.filelist[self.filelist['Split'] == split.upper()])) # type: ignore
        self.filelist = new_fl

        # Filter out videos that are not in the video folder
        self.fnames = [f+".avi" if not f.endswith(".avi") else f for f in self.filelist["FileName"].tolist()]
        self.fnames = [f for f in self.fnames if os.path.exists(os.path.join(self.video_folder_path, f))]

        # Split data if necessary:
        additional_info = ""
        if chunks > 1:
            elements_per_chunk = np.ceil(len(self.fnames) / chunks).astype(int)
            start_idx = chunk * elements_per_chunk
            end_idx = start_idx + elements_per_chunk
            self.fnames = self.fnames[start_idx:end_idx]
            additional_info = f". Chunk {chunk+1}/{chunks}."
        

        fps_dict = {f: v for f, v in zip(self.filelist["FileName"].tolist(), self.filelist["FPS"].tolist())}
        self.fps = [fps_dict[name[:-4]] for name in self.fnames]
        lvef_dict = {f: float(v) for f, v in zip(self.filelist["FileName"].tolist(), self.filelist["EF"].tolist())}
        self.lvef = [lvef_dict[name[:-4]] for name in self.fnames]

        # Create data loaders
        self.lazy_vloader = CachedVideoLoader(self.video_folder_path, deactivate=config.dataset.deactivate_cache)

        # Print info
        print(f"Loaded {len(self.fnames)} videos for {', '.join(self.splits)} split(s){additional_info}")
    
    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, i):

        # Get video
        fname = self.fnames[i] # video name
        video = self.lazy_vloader(fname) # load video from cache

        # Resample video to target fps
        original_fps = self.fps[i]
        video = zoom(video, (1, self.target_fps/original_fps, 1, 1), order=1)

        # Pad video to match target length if needed
        if video.shape[1] < self.videos_length:
            video = np.concatenate([video, np.zeros((3, self.videos_length-video.shape[1], 112, 112))], axis=1)

        # Randomly select a segment of the video
        start_index = 0 if self.videos_length == video.shape[1] else np.random.randint(0, video.shape[1]-self.videos_length)
        end_index = start_index + self.videos_length
        video = video[:, start_index:end_index, :, :]

        # Transform video - WARNING: video becomes a pytorch tensor
        video = self.transform(video) # -> # C T H W

        # Transform LVEF into an embedding
        lvef = self.lvef[i] # get lvef in range 0-100
        lvef = lvef / 100. # normalize to range 0-1
        lvef = torch.tensor(lvef)[None,None,...] # convert to tensor and add batch and channel dimensions

        # Add conditional image
        cond_idx = np.random.randint(0, video.shape[1])
        cond_frame = video[:, cond_idx, :, :]
        
        return video, lvef, cond_frame, fname

def get_seg_data(splits):
    path ='/lustre/project/ki-topml/jsiekiera/cf_data/'  # 'C:/Users/siekiera/Documents/git_repos/GradientSPN/' #
    data_path ='segment_paper_multiple_imgs.npz'#'sf_segmentation_frames_mask_09.npz'
    print(path + data_path)
    data = np.load(path + data_path)
    if "VAL" in splits:
        videos = data['val_min_max_frame']
        ef = data['val_ef_targets']
    elif "TRAIN" in splits:
        videos = data['train_min_max_frame']
        ef = data['train_ef_targets']
    else:
        videos = data['test_min_max_frame']
        ef = data['test_ef_targets']
    videos = np.transpose(videos, (0, 1, 4, 2, 3))/255.0
    ef/=100
    # goal (batchsize,3,2,112,112)
    return videos, ef

def get_data(splits):
    path = '/lustre/project/ki-topml/jsiekiera/cf_data/'  #'C:/Users/siekiera/Documents/git_repos/GradientSPN/' #
    data_path = 'sf_data.pkl'
    print(path + data_path)
    [train, test, val] = pkl.load(open(path + data_path, 'rb'))
    if "VAL" in splits:
        videos = val[0]
        ef = val[1]
    elif "TRAIN" in splits:
        videos = train[0]
        ef = train[1]
    else:
        videos = test[0]
        ef = test[1]
    videos = np.repeat(np.expand_dims(videos, axis=1), 3, axis=1) #(batchsize,3,112,112,2)
    videos = np.transpose(videos, (0, 1, 4, 2, 3)) #(batchsize,3,2,112,112)
    #ef *= 100
    return videos, ef

def get_data_CAMUS(splits):
    path =  '/lustre/project/ki-topml/jsiekiera/cf_data/' #'C:/Users/siekiera/Documents/curAIscid_data/camus_echonet_style/' #
    data_path = 'camus_echonet_style_mobi.npz'
    data = np.load(path + data_path)
    print(data["train_max_min_frame"].shape, data["train_max_min_masks"].shape, np.max(data["train_max_min_masks"]))
    split=splits[0].lower()
    target_type = 'bi'
    videos=data[split+"_max_min_frame"]
    ef=data[split+"_ef_"+target_type+"targets"]

    videos = np.repeat(videos, 3, axis=1)/255.0 #(batchsize,3,112,112,2)
    videos = np.transpose(videos, (0, 1, 4, 2, 3)) #(batchsize,3,2,112,112)
    ef /= 100
    print('CAMUS shape + params',videos.shape,np.min(videos),np.max(videos),np.min(ef),np.max(ef))
    return videos, ef


class EchoPair(Dataset):
    def __init__(self, config, splits=["TRAIN", "VAL", "TEST"], chunks=1, chunk=0,ds_name='CAMUS',show_case=False):

        # Arguments:
        self.config = config
        self.splits = splits if isinstance(splits, list) else [splits]

        # Data paths
        self.data_path = config.dataset.data_path
        self.video_folder_path = os.path.join(config.dataset.data_path, "Videos")

        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: torch.Tensor(x)),# / 255.0
            transforms.Resize(max(config.imagen.image_sizes)),
        ])

        if config.dataset.get("grayscale", False):
            self.transform.transforms.extend([
                transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # 3 64 112 112 => 64 3 112 112
                transforms.Grayscale(num_output_channels=3),  # type: ignore
                transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),  # 64 3 112 112 => 64 3 112 112
            ])

            # Load data points
        if ds_name == 'stanford_seg':
            self.videos, self.ef = get_seg_data(splits)
        elif ds_name == 'stanford':
            self.videos, self.ef = get_data(splits)
        elif ds_name == 'CAMUS':
            self.videos, self.ef = get_data_CAMUS(splits)
        else:
            print('DS not implemented', ds_name)
        if show_case:
            self.videos, self.ef =self.videos[:30], self.ef[:30]

        print(f"Loaded videos for {self.splits} split")

    def __len__(self):
        print('videos shape',self.videos.shape)
        return self.videos.shape[0]

    def __getitem__(self, i):

        # Get video
        lvef = float(self.ef[i])
        # Transform video - WARNING: video becomes a pytorch tensor
        video=self.videos[i]
        video = self.transform(video)  # -> # C T H W
        lvef = torch.tensor(lvef)[None, None, ...]  # convert to tensor and add batch and channel dimensions

        # Add conditional image
        cond_idx = np.random.randint(0, video.shape[1])
        cond_frame = video[:, cond_idx, :, :]
        #print('get video',i, lvef,video[0,0,0,:5])

        return video, lvef, cond_frame, str(i)