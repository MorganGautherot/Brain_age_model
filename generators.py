import numpy as np
import nibabel as nib
import math
from data_augmentation import coordinateTransformWrapperReg


class generator_mri_regression_norm_aug():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, norm=False, batch_size=8, shuffle=True):
        self.channels = list_path.shape[1]-1
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.idx = 0
        self.shuffle = shuffle
        self.norm = norm
        if self.shuffle :
            self.shuffle_list()

    def __norm__(self, batch):
        return np.array([ (x - np.min(x) )/ (np.max(x) - np.min(x)) for x in batch])
            
    def __getitem__(self, channels):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 	
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, channels]
        batch_img = [nib.load(path).get_data() for path in batch]

        if self.norm : 
            batch_img = self.__norm__(batch_img)

        return batch_img

    def __gettarget__(self):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, self.channels]
        return batch
    
    def __getbatch__(self):
        batch = np.stack([self.__getitem__(i) for i in range(self.channels)], axis=4)

        batch_x = []
        for T1 in batch[:, :, :, :, :self.channels] :                
                T1 = np.squeeze(T1,axis=3)
                T1_norm = coordinateTransformWrapperReg(T1)
                T1_norm = np.expand_dims(T1_norm, axis =3)
                batch_x.append(T1_norm)

        return np.array(batch_x), np.array(self.__gettarget__())
    
    def __iter__(self):
        if self.idx + 2 > self.len :
            self.idx = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.idx += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getbatch__()
            self.__iter__()
            yield (batch_input, batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len

class generator_mri_regression_norm():
    
    # Class is a dataset wrapper for better training performance
    def __init__(self, list_path, norm=False, batch_size=8, shuffle=True):
        self.channels = list_path.shape[1]-1
        self.list_path = list_path
        self.batch_size = batch_size
        self.len = math.ceil(len(self.list_path) / self.batch_size)
        self.idx = 0
        self.shuffle = shuffle
        self.norm = norm
        if self.shuffle :
            self.shuffle_list()

    def __norm__(self, batch):
        return np.array([ (x - np.min(x) )/ (np.max(x) - np.min(x)) for x in batch])
            
    def __getitem__(self, channels):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 	
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, channels]
        batch_img = [nib.load(path).get_data() for path in batch]

        if self.norm : 
            batch_img = self.__norm__(batch_img)

        return batch_img

    def __gettarget__(self):
        if (self.idx + 1) * self.batch_size > len(self.list_path) :
            idx_end = len(self.list_path)
        else :
            idx_end = (self.idx + 1) * self.batch_size 
            
        batch = self.list_path.iloc[self.idx * self.batch_size:idx_end, self.channels]
        return batch
    
    def __getbatch__(self):
        batch = np.stack([self.__getitem__(i) for i in range(self.channels)], axis=4)
        return np.array(batch[:, :, :, :, :self.channels]), np.array(self.__gettarget__())
    
    def __iter__(self):
        if self.idx + 2 > self.len :
            self.idx = 0
            if self.shuffle :
                self.shuffle_list()
        else :
            self.idx += 1
            
    # Generate flow of data
    def loader(self):
        # load data from somwhere with Python, and yield them    
        while True:
            batch_input, batch_output = self.__getbatch__()
            self.__iter__()
            yield (batch_input, batch_output)
            
    def shuffle_list(self):
        # shuffle the 
        self.list_path = self.list_path.sample(frac=1).reset_index(drop=True)
        
    def get_len(self):
        return self.len

        
        
