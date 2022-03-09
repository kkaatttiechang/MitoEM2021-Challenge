"""
Code inspired by Donglaiw @ https://github.com/donglaiw/MitoEM-challenge

    - h5_name : path to the directory from which the images will be read
    - h5_name : name of the H5 file to be created (follow the instructions in
                    https://mitoem.grand-challenge.org/Evaluation/ to name the 
                    files accordingly)
The H5 file should be saved in the directory where this script was called
"""

import os                                                                       
import h5py
import argparse
import numpy as np  
from os import path
from tqdm import tqdm
from scipy import ndimage
from skimage.io import imread
from PIL import ImageEnhance, Image
from skimage import measure, feature

def get_args():
    parser = argparse.ArgumentParser(description="H5 Conversion")
    parser.add_argument('--path', type=str, help='source path to the to-be-converted-h5 img dir')
    parser.add_argument('--name', type=str, help='name of the to-be-converted-h5 img dir')
    parser.add_argument('--saveto', type=str, help='dest path of the to-be-converted-h5 img dir')
    args = parser.parse_args()
    return args

def convert_to_h5(h5_path, h5_name, h5_saveto):
    img_shape = (4096, 4096)
    pred_ids = sorted(next(os.walk(h5_path))[2])  
    h5_name = '0_human_instance_seg_pred.h5'

    # Allocate memory for the predictions
    pred_stack = np.zeros((len(pred_ids),) + img_shape, dtype=np.int64)

    # Read all the images
    for n, id_ in tqdm(enumerate(pred_ids)):
        img = imread(os.path.join(h5_path, id_))
        pred_stack[n] = img
    
    # Apply connected components to make instance segmentation
    pred_stack = (pred_stack / 255).astype('int64')
    pred_stack, nr_objects = ndimage.label(pred_stack)
    print("Number of objects {}".format(nr_objects))

    # Create the h5 file (using lzf compression to save space)
    h5f = h5py.File(os.path.join(h5_saveto, h5_name), 'w')
    h5f.create_dataset('dataset_1', data=pred_stack, compression="lzf")
    h5f.close()

def main():
    args = get_args()
    save2 = os.getcwd()
    if args.saveto is not None: save2 = args.saveto 
    convert_to_h5(args.path, args.name, save2)

if __name__ == "__main__":
    main()