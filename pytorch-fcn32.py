import torch
import skimage.io as io
import numpy as np
import torchvision
from torchvision.models import vgg16
from torchvision.transforms import Compose,CenterCrop,Normalize,Scale
from torchvision.transforms import ToTensor,ToPILImage
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim import SGD,Adam
from utils.datasets import  VOC2012
from utils.transforms import Relabel,ToLabel,Palette
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import glob
import cv2
from PIL import Image



class fcn32(nn.Module):
    def __init__(self):
        super(fcn32,self).__init__()
        self.pretrained_model=vgg16(pretrained=True)
        features,classifiers=list(self.pretrained_model.features.children()),list(self.pretrained_model.classifier.children())

        features[0].padding=(100,100)
        self.features_map=nn.Sequential(*features)
        self.conv=nn.Sequential(nn.Conv2d(512,4096,7),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Conv2d(4096,4096,1),
                                nn.ReLU(inplace=True),
                                nn.Dropout()
                                )
        self.score_fr=nn.Conv2d(4096,21,1) 
        self.upscore=nn.ConvTranspose2d(21,21,64,32)

    def forward(self,x):
        x_size=x.size()
        pool=self.conv(self.features_map(x))
        score_fr=self.score_fr(pool)
        upscore=self.upscore(score_fr)
        return upscore[:,:,16:(16+x_size[2]),16:(16+x_size[3])]
 
fcn=fcn32()
fcn=fcn.cuda()
fcn=torch.nn.DataParallel(fcn,device_ids=[0,1])


data_dir="/home/chen/.keras/datasets/VOC2012/VOCdevkit/VOC2012/JPEGImages/"
train_txt_path="/home/chen/.keras/datasets/VOC2012/combined_imageset_train.txt"
label_dir="/home/chen/.keras/datasets/VOC2012/combined_annotations/"


input_transform=Compose([
                        Scale(256),
			CenterCrop(256),
                        ToTensor(),
                        Normalize([.485,.456,.406],[.229,.224,.225])
])

label_transform=Compose([
                        Scale(256),
			CenterCrop(256),
                        ToLabel(),
                        Relabel(255,0),
])


def train(model,batch_size,epoches):
    model.train()

    #weight=torch.ones(21)
    #weight[0]=0

    loader=DataLoader(VOC2012(data_dir,label_dir,train_txt_path,input_transform,label_transform),
                      num_workers=4,batch_size=batch_size,shuffle=True)
    
    #criterion=nn.NLLLoss2d(weight.cuda())
    criterion=nn.CrossEntropyLoss()
    optimizer=Adam(model.parameters(),1e-5)
    fps=24
    fourcc=cv2.VideoWriter_fourcc('M','J','P','G')
    vw=cv2.VideoWriter("demo/test.avi",fourcc,fps,(128*2,128))#same as show_image changing the width and height

    for epoch in range(1,epoches+1):
        for step,(images,labels) in enumerate(loader):
            images=images.cuda()
            labels=labels.cuda()

            optimizer.zero_grad()
            inputs=Variable(images)
            targets=Variable(labels)
            outputs=model(inputs)

            loss=criterion(outputs,targets[:,0])
            loss.backward()
            optimizer.step()
            if step%1==0:
                print(loss.data[0])
                show_demo(model,epoch,vw)
        save_weight(model,epoch)



def show_demo(model,i,videoWriter):
    raw_image=cv2.imread("demo/1.jpg")
    raw_image=cv2.resize(raw_image,(256,256))
    image=Image.open("demo/1.jpg")
    image=input_transform(image)
    image=image.cuda()
    label=model(Variable(image,volatile=True).unsqueeze(0))
    label=label.cpu()
    label=label.data.numpy()
    #print(label.shape)
    label=np.einsum('kilj->klji',label)
    label=np.squeeze(label)
    label=np.argmax(label,axis=-1)
    label=label.astype(np.uint8)
    label=Palette(label)
    show_image=np.concatenate([raw_image,label],axis=1)
    show_image=show_image.astype('uint8')
    show_image=cv2.resize(show_image,(128*2,128))
    videoWriter.write(show_image)

def save_weight(model,i):
    torch.save(model.state_dict(),f"outputs/{i}.pth")


train(fcn,2,40)

