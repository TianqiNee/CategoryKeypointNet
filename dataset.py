import os 
import cv2
import pickle
import torch
import numpy as np 
from torch.utils.data import Dataset
def decompress_from_positions(positions, image_shape, num_classes):
    """
    Decompress positions into a classification label image.
    
    Args:
        positions (list): List of positions for each class.
        image_shape (tuple): Shape of the output image (height, width).
        num_classes (int): Number of classes.
        
    Returns:
        np.ndarray: Classification label image with class indices as pixel values.
    """
    h, w = image_shape
    image = np.zeros((h, w), dtype=np.float32)
    
    for i in range(num_classes):
        image[positions[i][0], positions[i][1]] = i + 1.0  # Use class index as pixel value
    return image

def draw_umich_gaussian(heatmap, center, radius, k=1):
    """
    Draw a 2D Gaussian distribution on the heatmap.
    
    Args:
        heatmap (np.ndarray): Heatmap to draw on.
        center (tuple): Center coordinates (x, y) of the Gaussian.
        radius (int): Radius of the Gaussian distribution.
        k (float): Maximum value of the Gaussian (default: 1).
        
    Returns:
        np.ndarray: Updated heatmap with the Gaussian distribution.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def gaussian2D(shape, sigma=1):
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        shape (tuple): Shape of the Gaussian kernel (height, width).
        sigma (float): Standard deviation of the Gaussian distribution.
        
    Returns:
        np.ndarray: 2D Gaussian kernel.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h
class KeyPointDataset(Dataset):
    def __init__(self, image_dir, label_dir, size, transform=None, class_nums=3):
        """
        Initialize the KeyPointDataset.
        """
        self.image_paths = []
        self.cls_labels = []
        self.heatmap_labels = []
        base_names = os.listdir(image_dir)
        for base_name in base_names:
            self.image_paths.extend(sorted([os.path.join(image_dir, base_name, image) for image in os.listdir(os.path.join(image_dir, base_name)) if image.endswith(".jpg")]))
            self.cls_labels.extend(sorted([os.path.join(label_dir, base_name, "cls", label_heatmap) for label_heatmap in os.listdir(os.path.join(label_dir, base_name, "cls"))]))
            self.heatmap_labels.extend(sorted([os.path.join(label_dir, base_name, "heatmap", label_heatmap) for label_heatmap in os.listdir(os.path.join(label_dir, base_name, "heatmap"))]))
        self.class_nums = class_nums
        self.transform = transform
        self.new_size = size
        self.kernel = np.ones((5, 5), np.uint8)  # Create a 5x5 structuring element

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        """
        image = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        # label_heatmap = np.load(self.labels[idx])
        positions = pickle.load(open(self.cls_labels[idx], "rb"))
        label_cls = decompress_from_positions(positions, image.shape[:2], self.class_nums)
        new_label_cls = np.zeros((self.class_nums, image.shape[0], image.shape[1]))
        for i in range(self.class_nums):
            new_label_cls[i][label_cls == i + 1] = i + 1
        label_cls = new_label_cls

        with open(self.heatmap_labels[idx], "rb") as f:
            centers = pickle.load(f)
        heatmaps = []
        for center in centers:
            heatmaps.append(draw_umich_gaussian(np.zeros((image.shape[0], image.shape[1]), dtype=np.float32), center, 48))
        label_heatmap = np.stack(heatmaps, axis=0)
        
        if self.new_size is not None:
            image = cv2.resize(image, self.new_size, interpolation=cv2.INTER_LINEAR)

            new_label_heatmap = np.zeros((label_heatmap.shape[0], self.new_size[1], self.new_size[0]))
            for i in range(label_heatmap.shape[0]):
                new_label_heatmap[i] = cv2.resize(label_heatmap[i], self.new_size, interpolation=cv2.INTER_LINEAR)
                max_index = np.argmax(new_label_heatmap[i])  # Get the index of the maximum value
                max_position = np.unravel_index(max_index, new_label_heatmap[i].shape)
                new_label_heatmap[i][max_position] = 1. 
            label_heatmap = new_label_heatmap

            new_label_cls = np.zeros((self.class_nums, self.new_size[1], self.new_size[0]))
            mask = np.ones((self.new_size[1], self.new_size[0]))
            for i in range(self.class_nums):
                tmp_label = cv2.resize(label_cls[i], self.new_size, interpolation=cv2.INTER_LINEAR)
                tmp_label[image == 0] = 0
                tmp_label[mask == 0] = 0
                tmp_label[tmp_label != 0] = i + 1
                new_label_cls[i] = tmp_label
                mask[tmp_label != 0] = 0
            label_cls = new_label_cls
    
        image[image < 125] = 0.
        image[image > 125] = 255.
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float() 
        image = image.cuda()
        label_heatmap = torch.from_numpy(label_heatmap).cuda()
        label_cls = torch.from_numpy(label_cls).cuda()
        return image, label_heatmap, label_cls