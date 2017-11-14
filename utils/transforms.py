import numpy as np
import torch
from PIL import Image

palette_label = [[0,   0,   0],[128,   0,   0],[0, 128,   0],[128, 128,   0],
                 [0,   0, 128],[128,   0, 128],[0, 128, 128],[128, 128, 128],
                 [64,   0,   0],[192,   0,   0],[64, 128,   0],[192, 128,   0],
                 [64,   0, 128],[192,   0, 128],[64, 128, 128],[192, 128, 128],
                 [0,  64,   0],[128,  64,   0],[0, 192,   0],[128, 192,   0],[0,  64, 128]]


def Palette(image_gray):
    image_gray_size=image_gray.shape
    image_color=np.zeros((image_gray_size[0],image_gray_size[1],3),dtype=np.int32)
    for row in range(image_gray_size[0]):
        for col in range(image_gray_size[1]):
            image_color[row][col]=palette_label[image_gray[row][col]]
    return image_color


class Relabel:
    def __init__(self,olabel,nlabel):
        self.olabel=olabel
        self.nlabel=nlabel

    def __call__(self,tensor):
        assert isinstance(tensor,torch.LongTensor),'tensor needs to be LongTensor'
        tensor[tensor==self.olabel]=self.nlabel
        return tensor


class ToLabel:
    def __call__(self,image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)





