import os
import torch
import torchvision
from torch.utils.data import random_split
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
data_dir  = '/kaggle/input/garbage-classification/Garbage classification/Garbage classification'

classes = os.listdir(data_dir)
print(classes)
pip install imx500-converter[tf]

pip install tensorflow datasets huggingface_hub opencv-python-headless
from datasets import load_dataset

# Load dataset from Hugging Face
dataset = load_dataset("viola77data/recycling-dataset")
print(dataset)
