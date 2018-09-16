import numpy as np
import os
from multiprocessing import Pool
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

# Set RoI class and each colors

roi_class = ['Background',  'Hat', 'Hair', 'Glove', 'Sunglasses', 'Upper-clothes', 'Dress', 
             'Coat', 'Socks', 'Pants', 'Jumpsuits', 'Scarf', 'Skirt', 'Face', 'Left-arm',
             'Right-arm', 'Left-leg', 'Right-leg', 'Left-shoe', 'Right-shoe']

class_color = np.array((np.random.choice(range(255),len(roi_class)),
                        np.random.choice(range(255),len(roi_class)),
                        np.random.choice(range(255),len(roi_class)))).reshape(len(roi_class),3)

class_color[0]=[0,0,0]

# config GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"