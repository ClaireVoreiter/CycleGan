
# coding: utf-8

# Notebook use to create a classifier
# 
#     Impervious surfaces (RGB: 255, 255, 255)
#     Building (RGB: 0, 0, 255)
#     Low vegetation (RGB: 0, 255, 255)
#     Tree (RGB: 0, 255, 0)
#     Car (RGB: 255, 255, 0)
#     Clutter/background (RGB: 255, 0, 0)

# # Global Variable Used

# # Imports

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import rasterio

import random

import os, os.path

from tensorflow.python.keras.models import model_from_json

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix

import matplotlib.pyplot as plt

#import spectral

from scipy import sparse

import scipy.io as sio

from keras.utils import to_categorical

#import spectral.io.envi as envi

import cv2

import asyncio 

from scipy import io, misc
import imageio

from AdvGAN import dataset, models, utils, activations, cyclegan, layers, print_functions


# In[2]:


folder = '../Data_test/sentinel2/'
path, dirs, files = next(os.walk(folder))


# In[3]:


print(dirs)


# In[4]:


TILES_NAMES = [x[:-10] for x in files]
TILES_NAMES.sort()

TILES_NAMES_train = TILES_NAMES[:28]
TILES_NAMES_test = TILES_NAMES[28:]


# In[5]:


LABEL_DETAILS = [('No data', (0,0,0)),
          ('Impervious surfaces', (255, 255, 255)),
          ('Building', (0, 0, 255)),
          ('Low vegetation', (0, 255, 255)),
          ('Tree', (0,255,0)),
          ('Car', (255,255,0)),
          ('Clutter/background', (255,0,0))]

LABELS = [l[0] for l in LABEL_DETAILS]
N_CLASSES = len(LABELS) # Number of classes


# In[6]:


palette = {v: k[1] for v,k in enumerate(LABEL_DETAILS)}

invert_palette = {v: k for k, v in palette.items()}


# In[7]:


def convert_to_color(arr_2d, palette=palette):
    """ Numeric labels to RGB-color encoding """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d

def convert_from_color(arr_3d, palette=invert_palette):
    """ RGB-color encoding to grayscale labels """
    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


# In[8]:


def band_norm(img_, eps=1e-8):
    "normalize to [0-1]"
    img = np.asarray(img_, dtype='float32')
    for i in range(np.shape(img)[-1]):
        img[:,:,i] = (img[:,:,i] - np.min(img[:,:,i])) / (np.max(img[:,:,i]) +eps - np.min(img[:,:,i]))
    return img


# # Datasets

# In[9]:


translator = {'Crop': 0, 'Forest': 1, 'SeaLake': 2, 'Highway': 3, 'Industrial': 4, 'Pasture': 5,
              'Residential': 6, 'River': 7}


# In[10]:


folder = '../Data_test/sentinel2/'

path, dirs, files = next(os.walk(folder))
DATAS_Xt = []
LABEL_Xt = []

DATAS_Xt_light = []
LABEL_Xt_light = []

MANUEL = 650

for d in dirs:
    path_, dirs_, files_ = next(os.walk(path+d))
    DATAS_Xt += [np.array([band_norm(imageio.imread(path+d+'/'+x))]) for x in files_[:650]]
    LABEL_Xt += list(np.ones_like(files_[:650], dtype=int) * translator[d])

    DATAS_Xt_light += [np.array([band_norm(imageio.imread(path+d+'/'+x))]) for x in files_[:MANUEL]]
    LABEL_Xt_light += list(np.ones_like(files_[:MANUEL], dtype=int) * translator[d])
    
DATAS_Xt = np.reshape(DATAS_Xt, (-1, 64,64,13))
DATAS_Xt = (DATAS_Xt*2)-1

DATAS_Xt_light = np.reshape(DATAS_Xt_light, (-1, 64,64,13))
DATAS_Xt_light = (DATAS_Xt_light*2)-1


# In[11]:


folder = '../Data_test/nwpu/'

path, dirs, files = next(os.walk(folder))
DATAS_Xs = []
LABEL_Xs = []
for d in dirs:
    path_, dirs_, files_ = next(os.walk(path+d))
    DATAS_Xs += [np.array([imageio.imread(path+d+'/'+x)]) for x in files_[:650]]
    LABEL_Xs += list(np.ones_like(files_[:650], dtype=int) * translator[d])
    
DATAS_Xs = np.reshape(DATAS_Xs, (-1, 256, 256, 3))
DATAS_Xs = (DATAS_Xs/127.5 -1)


class Classifier:
    
    def  __init__(self, path):
        self.path = path
        
    def load_model(self):
        json_file = open('DA-GAN-master/model_nwpu.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = model_from_json(loaded_model_json)
        
    def load_weights(self):
        self.model.load_weights("DA-GAN-master/model_nwpu.h5")

    def predict(self, X):
        return self.model.predict(X)


# In[16]:


classifier = Classifier('model_nwpu')

# In[18]:

def gen_st(scope_name="generator", use_bias=False, use_sn=False,
           h=activations.lrelu(0.2), default_reuse=False, o=tf.tanh):
    
    def network(x, name='generator',reuse=False, trainig=True):

        with tf.variable_scope(name, reuse=reuse):

            net = layers.conv2D(x, 3, 64, 1, use_bias=use_bias, name='conv_init')
            net = h(net)
                
            input_stage = net
            
            for i in range(4):
                net = layers.conv2D(net, 3, 64, 1, use_bias=use_bias, name='block_conv'+str(i)+'_1')
                net = h(net)
                net = layers.conv2D(net, 3, 64, 1, use_bias=use_bias, name='block_conv'+str(i)+'_2')
                net = net + input_stage
                
 
            net = layers.conv2D(net, 3, 64, 1, use_bias=use_bias, name='conv_0')
            net = net + input_stage
        
            net = layers.conv2D(x, 3, 128, 1, use_bias=use_bias, name='conv1_1')
            net = h(net)
            net = layers.conv2D(net, 3, 128, 2, use_bias=use_bias, name='conv1_2')
            net = h(net)   
            
            net = layers.conv2D(net, 3, 128, 1, use_bias=use_bias, name='conv2_1')
            net = h(net)
            net = layers.conv2D(net, 3, 128, 2, use_bias=use_bias, name='conv2_2') 
            net = h(net)
            
            
            net = layers.conv2D(net, 3, 13, 1, use_bias=use_bias, name='conv_last')

            if(o != None):
                net = o(net)
                
            return net
    
    def build_net(x, name=scope_name, reuse=default_reuse, train=True):
        gen_img = network(x, name=name, reuse=reuse, trainig=train)
        return gen_img

    return build_net

def gen_ts(bs=64, scope_name="generator", use_bias=False, use_sn=False,
           h=activations.lrelu(0.2), default_reuse=False, o=tf.tanh):
    
    def network(x, name='generator',reuse=False, trainig=True):

        with tf.variable_scope(name, reuse=reuse):
            
            net = layers.conv2D(x, 3, 64, 1, use_bias=use_bias, name='conv_init')
            net = h(net)
                
            input_stage = net

            for i in range(4):
                net = layers.conv2D(net, 3, 64, 1, use_bias=use_bias, name='block_conv'+str(i)+'_1')
                net = h(net)
                net = layers.conv2D(net, 3, 64, 1, use_bias=use_bias, name='block_conv'+str(i)+'_2')
                net = net + input_stage
                
 
            net = layers.conv2D(net, 3, 64, 1, use_bias=use_bias, name='conv_0')
            net = net + input_stage
        
            net = layers.conv2D(x, 3, 128, 1, use_bias=use_bias, name='conv1_1')
            net = h(net)
            net = layers.deconv2D(net, 3, 128, 2, shape=[bs, 128, 128, 128],
                                      use_bias=use_bias, name='conv1_2')
            net = h(net)   
            
            net = layers.conv2D(net, 3, 128, 1, use_bias=use_bias, name='conv2_1')
            net = h(net)
            net = layers.deconv2D(net, 3, 128, 2, shape=[bs, 256, 256, 128],
                                      use_bias=use_bias, name='conv2_2')
            net = h(net)
            
            
            net = layers.conv2D(net, 3, 3, 1, use_bias=use_bias, name='conv_last')
            
            if(o != None):
                net = o(net)
            
            return net
    
    def build_net(x, name=scope_name, reuse=default_reuse, train=True):
        gen_img = network(x, name=name, reuse=reuse, trainig=train)
        return gen_img

    return build_net

# In[19]:


# Batch size
bs = 16

# B is RGB
generator_t = gen_st(scope_name="generator_t")

# A is Greyscale
generator_s = gen_ts(bs=bs, scope_name="generator_s")

# Same architecture for both
discriminator_t = models.make_d_conv(
                    hiddens_dims=[32,64,128,256],
                    scope_name="discriminator_t",
                    h=activations.lrelu(0.2),
                    use_sn=False,
                    keep_prob=0.8
                    )

discriminator_s = models.make_d_conv(
                    hiddens_dims=[32,64,128,256],
                    scope_name="discriminator_s",
                    h=activations.lrelu(0.2),
                    use_sn=False,
                    keep_prob=0.8
                    )


# In[20]:


# G_OPTIMIZER
g_optim = {'name': 'RMSProp', 'learning_rate': 5e-5}

# D_OPTIMIZER
d_optim = {'name': 'RMSProp', 'learning_rate': 1e-4}

# Build option for CFG (options not set in cfg are set with default values)
gan = cyclegan.make_gan('../DA-GAN-master/cfg/DA_test.cfg')

gan.set_generator_s(generator_s)
gan.set_generator_t(generator_t)

gan.set_discriminator_s(discriminator_s)
gan.set_discriminator_t(discriminator_t)

gan.set_classifier_s(classifier, 8)

nb_class = len(dirs)

mask = np.zeros((len(LABEL_Xt),1))

n = int((len(LABEL_Xt) / nb_class) / 2)
off = 0
for i in range(nb_class):
    off = int((len(LABEL_Xt) / nb_class) * i)
    mask[off:off+n] = 1

data = dataset.Cycle_Dataset_tf(DATAS_Xs, DATAS_Xt, to_categorical(LABEL_Xs), to_categorical(LABEL_Xt), mask, batch_size=bs)
data.shuffle()

gan.ma_gamma = 1e-6
gan.set_dataset(data)
gan.set_batch_size(bs, mini_batch_max=16)

# Need to build before train
gan.build_model(reset_graph=True)
print(data.shape())


gan.train(restore=True, print_method=print_functions.save_cycle_samples_and_loss(nb_channels_s=3,
                                                                nb_channels_t=13, nb_img=8, bands=[4,3,2],
                                                                size_s=256, size_t=64), data_pretrain= dataset.Dataset_simple(DATAS_Xt_light, to_categorical(LABEL_Xt_light)))


# In[ ]:


gan.data.shape()


# In[ ]:


np.shape(LABEL_Xs)

