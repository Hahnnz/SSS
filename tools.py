from config import *
import numpy as np
import os, glob, cv2

def masking(answer, Human_only=False):
    mask = np.zeros(list(answer.shape[:3])+[3,])
    batch, width, height = answer.shape[:3]
    for b in range(batch):
        for w in range(width):
            for h in range(height):
                if not Human_only : mask[b,w,h] = class_color[answer[b,w,h]]
                elif Human_only : mask[b,w,h] = [0,0,0] if mask[b,w,h,0] == 0 else [255,255,255]
    return np.array(mask, np.uint8)

def explore_dir(dir,count=0,f_extensions=None):
    if count==0:
        global n_dir, n_file, filenames, filelocations
        n_dir=n_file=0
        filenames=list()
        filelocations=list()

    for img_path in sorted(glob.glob(os.path.join(dir,'*' if f_extensions is None else '*.'+f_extensions))):
        if os.path.isdir(img_path):
            n_dir +=1
            explore_dir(img_path,count+1)
        elif os.path.isfile(img_path):
            n_file += 1
            filelocations.append(img_path)
            filenames.append(img_path.split("/")[-1])
    return np.array((filenames,filelocations))

def separate_batches(img,label,batch_size):
    img_batches = list()
    label_batches = list()
        
    if(len(img)%batch_size==0):
        num_batches = len(img)//batch_size
        for nb in range(num_batches):
            img_batches.append(img[batch_size*nb:batch_size*(nb+1)])
            label_batches.append(label[batch_size*nb:batch_size*(nb+1)])
    else:
        num_batches = (len(img)//batch_size)+1
        for nb in range(num_batches):
            img_batches.append(img[batch_size*nb:batch_size*(nb+1)])
            label_batches.append(label[batch_size*nb:batch_size*(nb+1)])
            if(nb==num_batches-1):
                img_batches.append(img[batch_size*nb:])
                label_batches.append(label[batch_size*nb:])
                
    return img_batches,label_batches

def im_and_roi_read(path):
    img, roi = cv2.imread(path[0]), cv2.imread(path[1])
    return cv2.resize(img,(256,256)), cv2.resize(roi,(256,256))