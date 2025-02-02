# -*- coding: utf-8 -*-
# @Time    : 2020/12/11 22:05
# @Author  : Mingxing Li
# @FileName: iteration_eval.py
# @Software: PyCharm

"""
This script looks very complicated, but in fact most of them are default parameters.
There are some important parameters:
SYSTEM.NUM_GPUS
SYSTEM.NUM_CPUS
INFERENCE.INPUT_SIZE: Although the training size is only 32 × 256 × 256, we have empirically found that D=100 is better.
                      (Thanks to the fully convolutional network, the input size is variable)
INFERENCE.STRIDE
INFERENCE.PAD_SIZE
INFERENCE.AUG_NUM: 0 is faster
"""

import subprocess

def cal_infer(root_dir, model_id):
    """
    If you have enough resources, you can use this function during training. 
    Confirm that this line is open. 
    https://github.com/Limingxing00/MitoEM2021_Challenge/blob/dddb388a4aab004fa577058b53c39266e304fc03/connectomics/engine/trainer.py#L423
    """

    command = "{} {}/scripts/main.py --config-file\
                {}/configs/MitoEM/MitoEM-R-BC.yaml\
                --inference\
                --do_h5\
                --checkpoint\
                {}/outputs/dataset_output/checkpoint_{:06d}.pth.tar\
                --opts\
                SYSTEM.ROOTDIR\
                {}\
                SYSTEM.NUM_GPUS\
                2\
                SYSTEM.NUM_CPUS\
                8\
                DATASET.DATA_CHUNK_NUM\
                [1,1,1]\
                INFERENCE.SAMPLES_PER_BATCH\
                2\
                INFERENCE.INPUT_SIZE\
                [100,256,256]\
                INFERENCE.OUTPUT_SIZE\
                [100,256,256]\
                INFERENCE.STRIDE\
                [1,256,256]\
                INFERENCE.PAD_SIZE\
                [0,256,256]\
                INFERENCE.AUG_NUM\
                0\
            ".format(which_python, root_dir, root_dir, root_dir, model_id, root_dir)

    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")

    command = "{} {}/connectomics/utils/evaluation/evaluate.py \
                 -gt \
                 {}/0_human_instance_seg_pred.h5 \
                 -p \
                 {}/outputs/inference_output/{:06d}_out_100_256_256_aug_0_pad_0.h5 \
             -o {}{:06d}".format(which_python, root_dir, root_dir, model_id, root_dir, model_id)

    out = subprocess.run(command, shell=True)
    print(command, "\n |-------------| \n", out, "\n |-------------| \n")



if __name__=="__main__":
    """
    Please note to change the gt file!
    My gt file is in:
    /braindat/lab/limx/MitoEM2021/MitoEM-H/MitoEM-H/0_human_instance_seg_pred.h5
    """
    # change the "start_epoch" and "end_epoch"  to infer "root_dir/model"
    start_epoch, end_epoch = 297000, 297000
    step_epoch = 2500
    model_id = range(start_epoch, end_epoch+step_epoch, step_epoch)

    root_dir = "." # WRITE PATH TO ROOT 
    global which_python 
    which_python = "." # WRITE PATH TO PYTHON, i.e. /opt/conda/bin/python
    #"/braindat/lab/limx/MitoEM2021/CODE/HUMAN/rsunet_retrain_297000_v2/"


    # validation stage: output h5
    # test stage: don't output h5
    for i in range(len(model_id)):  
        command = "{} {}/scripts/main.py --config-file\
                {}/configs/MitoEM/MitoEM-R-BC.yaml\
                --inference\
                --do_h5\
                --checkpoint\
                {}/outputs/dataset_output/checkpoint_{:06d}.pth.tar\
                --opts\
                SYSTEM.ROOTDIR\
                {}\
                SYSTEM.NUM_GPUS\
                2\
                SYSTEM.NUM_CPUS\
                8\
                DATASET.DATA_CHUNK_NUM\
                [1,1,1]\
                INFERENCE.SAMPLES_PER_BATCH\
                4\
                INFERENCE.INPUT_SIZE\
                [100,256,256]\
                INFERENCE.OUTPUT_SIZE\
                [100,256,256]\
                INFERENCE.STRIDE\
                [1,128,128]\
                INFERENCE.PAD_SIZE\
                [0,128,128]\
                INFERENCE.AUG_NUM\
                0\
                ".format(which_python, root_dir, root_dir, root_dir, model_id[i], root_dir)

        out = subprocess.run(command, shell=True)
        print(command, "\n |-------------| \n", out, "\n |-------------| \n")



        command = "{} {}/connectomics/utils/evaluation/evaluate.py \
             -gt \
             {}/0_human_instance_seg_pred.h5 \
             -p \
             {}/outputs/inference_output/{:06d}_out_100_256_256_aug_0_pad_0.h5 \
                 -o {}{:06d}".format(which_python, root_dir, root_dir, root_dir, model_id[i], root_dir, model_id[i])

        out = subprocess.run(command, shell=True)
        print(command, "\n |-------------| \n", out, "\n |-------------| \n")
