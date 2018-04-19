# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 21:58:46 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""

import numpy as np
from scipy.misc import imread,imresize
import os
import time
path='C:\\Users\\resu\\Desktop\\SRCNN\\'
data1=os.listdir('database')
data2=os.listdir('database1')
data3=os.listdir('database2')
f=33
stride=14
scaling_factor=3
image_high=np.zeros(3267).reshape(1,33,33,3)
image_low=np.zeros(3267).reshape(1,33,33,3)
def crop_image(count,image_high,image_low):
    t0=time.time()
    for i in range(len(data1)):
        I=imread('database\\'+data1[i])
        h,w,c=I.shape
        I_low=imresize(I,(int(h/scaling_factor),int(w/scaling_factor)),'bicubic')
        I_low=imresize(I_low,(h,w),'bicubic')
        x=int(np.floor((h-f)/stride)+1)
        y=int(np.floor((w-f)/stride)+1)
        for p in range(x):
            for q in range(y):
                im=I[p*stride:p*stride+33,q*stride:q*stride+33,:].reshape(1,33,33,3)
                im_low=I_low[p*stride:p*stride+33,q*stride:q*stride+33,:].reshape(1,33,33,3)
                image_high=np.concatenate((image_high,im),axis=0)
                image_low=np.concatenate((image_low,im_low),axis=0)
                count=count+1
                if count%10000==0:
                    print('we have already cropped {} pictures'.format(count))
                    if count%50000==0:
                         np.save('image_high'+str(int(count/50000))+'.npy',image_high[1:,:,:,:])
                         np.save('image_low'+str(int(count/50000))+'.npy',image_low[1:,:,:,:])
                         image_high=np.zeros(3267).reshape(1,33,33,3)
                         image_low=np.zeros(3267).reshape(1,33,33,3)
                         print('database'+str(i))
    for i in range(len(data2)):
        I=imread('database1\\'+data2[i])
        h,w,c=I.shape
        I_low=imresize(I,(int(h/scaling_factor),int(w/scaling_factor)),'bicubic')
        I_low=imresize(I_low,(h,w),'bicubic')
        x=int(np.floor((h-f)/stride)+1)
        y=int(np.floor((w-f)/stride)+1)
        for p in range(x):
            for q in range(y):
                im=I[p*stride:p*stride+33,q*stride:q*stride+33,:].reshape(1,33,33,3)
                im_low=I_low[p*stride:p*stride+33,q*stride:q*stride+33,:].reshape(1,33,33,3)
                image_high=np.concatenate((image_high,im),axis=0)
                image_low=np.concatenate((image_low,im_low),axis=0)
                count=count+1
                if count%10000==0:
                    print('we have already cropped {} pictures'.format(count))
                    if count%50000==0:
                         np.save('image_high.npy'+str(count/50000),image_high[1:,:,:,:])
                         np.save('image_low.npy'+str(count/50000),image_low[1:,:,:,:])
                         image_high=np.zeros(3267).reshape(1,33,33,3)
                         image_low=np.zeros(3267).reshape(1,33,33,3)
                         print('database1'+str(i))
    for i in range(len(data3)):
        I=imread('database2\\'+data3[i])
        h,w,c=I.shape
        I_low=imresize(I,(int(h/scaling_factor),int(w/scaling_factor)),'bicubic')
        I_low=imresize(I_low,(h,w),'bicubic')
        x=int(np.floor((h-f)/stride)+1)
        y=int(np.floor((w-f)/stride)+1)
        for p in range(x):
            for q in range(y):
                im=I[p*stride:p*stride+33,q*stride:q*stride+33,:].reshape(1,33,33,3)
                im_low=I_low[p*stride:p*stride+33,q*stride:q*stride+33,:].reshape(1,33,33,3)
                image_high=np.concatenate((image_high,im),axis=0)
                image_low=np.concatenate((image_low,im_low),axis=0)
                count=count+1
                if count%10000==0:
                    print('we have already cropped {} pictures'.format(count))
                    if count%50000==0:
                         np.save('image_high.npy'+str(count/50000),image_high[1:,:,:,:])
                         np.save('image_low.npy'+str(count/50000),image_low[1:,:,:,:])
                         image_high=np.zeros(3267).reshape(1,33,33,3)
                         image_low=np.zeros(3267).reshape(1,33,33,3)
                         print('database2'+str(i))
    np.save('image_high.npy',image_high[1:,:,:,:])
    np.save('image_low.npy',image_low[1:,:,:,:])
    t1=time.tiem()
    return count,t1-t0

count,t=crop_image(0,image_high,image_low)
print('the number of pictures: {}'.format(count))
print('the time of prepare data: {}h'.format(t/3600))