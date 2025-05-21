import os 
import cv2
import numpy as np
import torch 
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import KeyPointDataset
from model import CategoryKeypointNet
from evaluation import*


# Object name and image size

# object_name = "strip"
# size = (1344,96)

# object_name = "rectangle"
# size = (480,240) 

object_name = "square"
size = (320,320)


# Directory paths
image_dir = f"data/{object_name}/test_data"
label_dir = f"data/{object_name}/test_label"
weight_dir = f"data/{object_name}/output"
save_dir = f"test/{object_name}"
os.makedirs(save_dir, exist_ok=True)

# Data loader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.], std=[1.])
])

dataset = KeyPointDataset(image_dir, label_dir, size=size, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Model initialization
model = CategoryKeypointNet(n_channels=1, n_classes=3, bilinear=True).cuda()  # Assume there are 10 keypoints

# Load the model weights
model.load_state_dict(torch.load(f'path/square_final.pth'))

# Set the model to evaluation mode
model.eval()  

# Evaluate the detection performance
evaluate_detection(model, dataloader, threshold=20)
