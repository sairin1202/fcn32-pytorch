import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
from .transforms import Palette

def load_image(file):
    return Image.open(file)

def get_images(filename):
    image_names=np.loadtxt(filename,dtype=np.str)
    return image_names,len(image_names)


class VOC2012(Dataset):
    def __init__(self,data_dir,label_dir,train_file,input_transform=None,label_transform=None):
        self.data_dir=data_dir
        self.label_dir=label_dir
        self.input_transform=input_transform
        self.label_transform=label_transform
        self.train_file=train_file
        self.train_image_names,self.train_length=get_images(train_file)

    def __getitem__(self,index):

        filename=self.train_image_names[index][2:-1]
        with open(self.data_dir+str(filename)+'.jpg','rb') as f:
            image=load_image(f).convert('RGB')
        with open(self.label_dir+str(filename)+'.png','rb') as f:
            label=load_image(f).convert('P')

        if self.input_transform is not None:
            image=self.input_transform(image)
        if self.label_transform is not None:
            label=self.label_transform(label)
        #label_numpy=label.numpy()
        #image_numpy=image.numpy()
        #label_numpy=label_numpy.astype(np.uint8)
        #image_numpy=image_numpy.astype(np.uint8)
        #label_numpy=np.einsum('ilj->lji',label_numpy)
        #image_numpy=np.einsum('ilj->lji',image_numpy)
        #label_numpy=np.squeeze(label_numpy)
        #print(label_numpy.shape)
        #print(image_numpy.shape)
        #io.imsave(f"demo/{index}.png",Palette(label_numpy))
        #io.imsave(f"demo/{index}.jpg",image_numpy)	
        return image,label

    def __len__(self):
        return len(self.train_image_names)
