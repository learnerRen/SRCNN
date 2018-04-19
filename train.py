# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 23:54:45 2018

@author: Oliver Ren

e-mail=OliverRensu@gmail.com
"""
'''
import read_cifar100
cirfar100=read_cifar100.unpickle('cifar-100-python\\train')
cirfar100=cirfar100[b'data']
'''
import os 
import tensorflow as tf
import model
import numpy as np
import time
batch_size=128
#learning_rate_base=0.001
#learning_rate_decay=0.99
epoch=1000000
model_save_path="./path/model/"
model_name="model.ckpt"
im_size_low=model.im_size-model.f1-model.f2-model.f3+3
center_left=int((model.f1+model.f2+model.f3-3)/2)
center_right=center_left+im_size_low
pic_high=np.load('image_high1.npy')
pic_low=np.load('image_low1.npy')
'''
pic_high2=np.load('image_high2.npy')
pic_low2=np.load('image_low2.npy')
pic_high=np.concatenate((pic_high1,pic_high2),axis=0)
pic_low=np.concatenate((pic_low1,pic_low2),axis=0)
'''

pic_num=pic_high.shape[0]
if pic_num !=pic_high.shape[0]:
    print("wrong")
def nextBatch(count):
    if count+batch_size>pic_num:
        xs=pic_low[count:pic_num,:,:,:]
        ys=pic_high[count:pic_num,center_left:center_right,center_left:center_right,:]
        count=(count+batch_size)%pic_num
        xs=np.concatenate((xs,pic_low[0:count,:,:,:]),axis=0)
        ys=np.concatenate((ys,pic_high[0:count,center_left:center_right,center_left:center_right,:]),axis=0)
    else:
        xs=pic_low[count:count+batch_size,:,:,:]
        ys=pic_high[count:count+batch_size,center_left:center_right,center_left:center_right,:]
    count=count+batch_size
    return xs,ys,count%pic_num
def train():
    t0=time.time()
    x=tf.placeholder(tf.float32,
                     [batch_size,
                      model.im_size,
                      model.im_size,
                      model.n],
                     name='x-input')
    y=tf.placeholder(tf.float32,
                     [batch_size,
                      im_size_low,
                      im_size_low,
                      model.n],
                     name='y-output')
    y_hat=model.inference(x)
    global_step=tf.Variable(0,trainable=False)
    loss=tf.reduce_mean(tf.square(y-y_hat),name='mse')
    '''
    learning_rate=tf.train.exponential_decay(
            learning_rate_base,
            global_step,
            pic_num/batch_size,
            learning_rate_decay)
    '''
    count=0
    train_op=tf.train.AdamOptimizer(0.00001).minimize(loss,global_step=global_step)
    saver=tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(epoch):
            xs,ys,count=nextBatch(count)
            _,step=sess.run([train_op,global_step],feed_dict={x:xs,y:ys})
            if i%100000==0:
                loss_value,step=sess.run([loss,global_step],feed_dict={x:xs,y:ys})
                print("After {} training step(s),loss is {}".format(i,loss_value))
                saver.save(sess,os.path.join(model_save_path,model_name),global_step=global_step)
    t1=time.time()
    return t1-t0
        #y_value=sess.run(y_hat,feed_dic={x:pic_test})
        #imsave('new_pic',y_value)
'''        
def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)
    
if __name__=='__main__':
    tf.app.run()        
'''
t=train()
print(t)
print('train time: {}h'.format(t/3600))