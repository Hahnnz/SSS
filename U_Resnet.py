import tensorflow as tf
import numpy as np
from layers import *

def identity_block(data, ksize, filters, stage, block, use_bias=True):
    suffix=str(stage)+block+"_branch"
    conv_name_base = "res"+suffix
    bn_name_base = "bn"+suffix
    
    filter1, filter2, filter3 = filters
    
    data_bn_relu = tf.nn.relu(tf.contrib.layers.batch_norm(data))
    
    conv1 = conv(data_bn_relu, ksize, filter1, ssize=1, padding="SAME", conv_name=conv_name_base+"2a",
                  bn_name=bn_name_base+"2a", use_bias=use_bias, bn=True, act=False)
    conv2 = conv(conv1, ksize, filter2, ssize=1, padding="SAME",conv_name=conv_name_base+"2b",
                  bn_name=bn_name_base+"2b", use_bias=use_bias, bn=True, act=False)
    conv3 = conv(conv2, ksize, filter3, ssize=1, padding="SAME",conv_name=conv_name_base+"2c",
                  bn_name=bn_name_base+"2c", use_bias=use_bias, bn=False, act=False)
    if int(data.shape[-1])!=filter3:
        shortcut = conv(data, 1, filter3, ssize=1, padding="SAME",
                        conv_name=conv_name_base+"shortcut", use_bias=False, bn=False, act=False)
    else :
        shortcut = data
    addx_h = tf.contrib.layers.batch_norm(tf.add(conv3, shortcut))
    
    return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

def conv_block(data, kernel_size, filters, stage, block, ssize, use_bias=True):
    suffix=str(stage)+block+"_branch"
    conv_name_base = "res"+suffix
    bn_name_base = "bn"+suffix
    
    data_bn_relu = tf.nn.relu(tf.contrib.layers.batch_norm(data))
    
    conv1 = conv(data_bn_relu, kernel_size, filters[0], ssize=ssize, padding="SAME",conv_name=conv_name_base+"2a",
                 bn_name=bn_name_base+"2a",use_bias=use_bias,bn=True,act=False)
    conv2 = conv(conv1, kernel_size, filters[1], ssize=1, padding="SAME",conv_name=conv_name_base+"2b",
                 bn_name=bn_name_base+"2b",use_bias=use_bias,bn=True,act=False)
    conv3 = conv(conv2, kernel_size, filters[2], ssize=1, padding="SAME",conv_name=conv_name_base+"2c",
                 bn_name=bn_name_base+"2c",use_bias=use_bias,bn=False,act=False)
    
    if int(data.shape[-1])!=filters[2]:
        shortcut = conv(data, 1, filters[2], ssize=1, padding="SAME",
                        conv_name=conv_name_base+"shortcut", use_bias=False, bn=False, act=False)
    else :
        shortcut = data
    addx_h = tf.contrib.layers.batch_norm(tf.add(conv3, shortcut))
    
    return tf.nn.relu(addx_h, name="res"+str(stage)+block+"_out")

class U_Resnet:
    def __init__(self,input_shape,output_shape,num_classes,gpu_memory_fraction=None):
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.num_classes = num_classes
        
        self.x = tf.placeholder(tf.float32, (None,) + self.input_shape, name='X')
        self.y = tf.placeholder(tf.int64, shape=(None,)+ self.output_shape, name='y')
        self.keep_prob = tf.placeholder(tf.float32)
        
        config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
        if gpu_memory_fraction is None:
            config.gpu_options.allow_growth = True
        else:
            config.gpu_options.per_process_gpu_memory_fraction = gpu_memory_fraction
            
        self.__create() 
        self.sess = tf.Session(config=config)
        self.graph = tf.get_default_graph()
    
    def __create(self):
        # Down Block 1
        convblock_1 = conv_block(self.x,3,[64,64,64], stage=1, block="a", ssize=1)
        id_block_2 = identity_block(convblock_1, 3, [64,64,64], stage=1, block="b")
        id_block_3 = identity_block(id_block_2, 3, [64,64,64], stage=1, block="c")
        pool1 = max_pooling(id_block_3, 3, 2)

        # Down Block 2
        convblock_4 = conv_block(pool1,3,[128,128,128], stage=2, block="a", ssize=1)
        id_block_5 = identity_block(convblock_4, 3, [128,128,128], stage=2, block="b")
        id_block_6 = identity_block(id_block_5, 3, [128,128,128], stage=2, block="c")
        pool2 = max_pooling(id_block_6, 3, 2)
        
        # Down Block 3
        convblock_7 = conv_block(pool2,3,[256,256,256], stage=3, block="a", ssize=1)
        id_block_8 = identity_block(convblock_7, 3, [256,256,256], stage=3, block="b")
        id_block_9 = identity_block(id_block_8, 3, [256,256,256], stage=3, block="c")
        pool3 = max_pooling(id_block_9, 3, 2)
        
        # Down Block 4
        convblock_10 = conv_block(pool3,3,[512,512,512], stage=4, block="a", ssize=1)
        id_block_11 = identity_block(convblock_10, 3, [512,512,512], stage=4, block="b")
        id_block_12 = identity_block(id_block_11, 3, [512,512,512], stage=4, block="c")
        drop4 = dropout(id_block_12, name='drop4', ratio=self.keep_prob)
        pool4 = max_pooling(drop4, 3, 2)

        # Down Block 5
        convblock_13 = conv_block(pool4,3,[1024,1024,1024], stage=5, block="a", ssize=1)
        id_block_14 = identity_block(convblock_13, 3, [1024,1024,1024], stage=5, block="b")
        id_block_15 = identity_block(id_block_14, 3, [1024,1024,1024], stage=5, block="c")
        drop5 = dropout(id_block_15, name='drop5', ratio=self.keep_prob)
        
        # Up Block 4
        self.deconv6 = deconv(drop5, ksize=3, filters=512, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="up6", bn_name='up6', bn=True)
        self.concat6 = tf.concat((drop4,self.deconv6),axis=3, name='concat6')
        convblock_16 = conv_block(self.concat6,3,[512,512,512], stage=6, block="a", ssize=1)
        id_block_17 = identity_block(convblock_16, 3, [512,512,512], stage=6, block="b")
        id_block_18 = identity_block(id_block_17, 3, [512,512,512], stage=6, block="c")

        # Up Block 3
        self.deconv7 = deconv(id_block_18, ksize=3, filters=256, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="up7", bn_name='up7', bn=True)
        self.concat7 = tf.concat((id_block_9,self.deconv7),axis=3, name='concat7')
        convblock_19 = conv_block(self.concat7,3,[256,256,256], stage=7, block="a", ssize=1)
        id_block_20 = identity_block(convblock_19, 3, [256,256,256], stage=7, block="b")
        id_block_21 = identity_block(id_block_20, 3, [256,256,256], stage=7, block="c")

        # Up Block 2
        self.deconv8 = deconv(id_block_21, ksize=3, filters=128, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="up8", bn_name='up8', bn=True)
        self.concat8 = tf.concat((id_block_6,self.deconv8),axis=3, name='concat8')
        convblock_22 = conv_block(self.concat8,3,[128,128,128], stage=8, block="a", ssize=1)
        id_block_23 = identity_block(convblock_22, 3, [128,128,128], stage=8, block="b")
        id_block_24 = identity_block(id_block_23, 3, [128,128,128], stage=8, block="c")

        # Up Block 1
        self.deconv9 = deconv(id_block_24, ksize=3, filters=64, ssize=2, use_bias=True, padding='SAME',
                              deconv_name="up9", bn_name='up9', bn=True)
        self.concat9 = tf.concat((id_block_3,self.deconv9),axis=3, name='concat9')
        convblock_25 = conv_block(self.concat9,3,[64,64,64], stage=9, block="a", ssize=1)
        id_block_26 = identity_block(convblock_25, 3, [64,64,64], stage=9, block="b")
        id_block_27 = identity_block(id_block_26, 3, [64,64,64], stage=9, block="c")

        # Scoring
        self.score = conv(id_block_27, ksize=1, filters=self.num_classes, ssize=1, use_bias=True, padding='SAME',
                          conv_name='score', bn=False, act=True)
