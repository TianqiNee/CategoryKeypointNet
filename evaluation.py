import numpy as np
import torch
from torch.utils.data import DataLoader
import os
from torchvision import transforms
from scipy.spatial import distance_matrix
import logging
from sklearn.metrics import jaccard_score, accuracy_score

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def ensure_single_point(label):
    """
    Ensure that there is only one annotated point in each channel.
    If there are multiple points, select the first point as the annotated point.
    """
    for channel in range(label.shape[0]):
        points = np.argwhere(label[channel] > 0 )
        
        if len(points) > 1:
            label[channel] = np.zeros_like(label[channel])
            label[channel][points[0][0], points[0][1]] = 1
    return label

def get_label_points(label):
    """
    Get the coordinates of the annotated points.
    """
    label_points = []
    for channel in range(label.shape[0]):
        channel_points = np.argwhere(label[channel] > 0)
        if len(channel_points) != 1:
            logging.warning(f"Channel {channel} does not contain exactly one point.")
            continue
        label_points.append(channel_points[0])
    return label_points

def get_detected_points(outputs):
    """
    Get the coordinates of the detected points.
    """
    detected_points = []
    for channel in range(outputs.shape[0]):
        max_index = np.argmax(outputs[channel])
        detected_point = np.unravel_index(max_index, outputs[channel].shape)
        detected_points.append(detected_point)
    return detected_points

def calculate_matching_accuracy(label_points, detected_points, threshold):
    """
    Calculate the matching accuracy between label points and detected points.
    """
    label_tensor = torch.tensor(np.stack(label_points)).float()   # shape: [N1, 2]
    detected_tensor = torch.tensor(np.stack(detected_points)).float()  # shape: [N2, 2]

    n1, n2 = label_tensor.shape[0], detected_tensor.shape[0]
    dists = torch.cdist(label_tensor, detected_tensor, p=2)  # [n1, n2]

    matched_count = 0
    matched_index = []
    matched_errors = []
    all_errors = []

    for i in range(min(n1, n2)):
        dist = torch.norm(label_tensor[i] - detected_tensor[i], p=2).item()
        all_errors.append(dist)
        if dist <= threshold:
            matched_count += 1
            matched_index.append((i, i))  # Index pairs of successful sequential matches
            matched_errors.append(dist)

    return matched_count, matched_index, matched_errors, all_errors

def evaluate_detection(model, dataloader, threshold=20):
    """
    Evaluate the detection performance of the model.
    """
    matched_localization_error = 0
    all_localization_error = 0
    matched_counts = 0
    counts = 0
    for image, label_heatmap, label_cls, keypoints_list in dataloader:
        image = image.cuda()
        scores = model(image, keypoints_list)
        # Point localization evaluation
        scores = scores.squeeze(0).cpu().detach().numpy()
        keypoints_list = keypoints_list.squeeze(0).cpu().detach().numpy()
        
        # Get the positions of the annotated points
        label_heatmap = label_heatmap.squeeze(0)
        tmp_label_points = torch.nonzero(label_heatmap == 1)
        assert len(tmp_label_points) == 2
        label_points = [tmp_label_points[0][1:].cpu().numpy(), tmp_label_points[1][1:].cpu().numpy()]

        # Get the positions of the detected points
        detected_points = []
        for c in range(scores.shape[-1]):
            index = np.argmax(scores[:, c])
            detected_points.append(keypoints_list[index])

        if len(detected_points) != 2:
            logging.warning("Detected points do not match the expected number.")
            continue

        matched_count, matched_index, matched_errors, all_errors = calculate_matching_accuracy(label_points, detected_points, threshold)

        matched_localization_error += np.sum(matched_errors)
        all_localization_error += np.sum(all_errors)
        matched_counts += matched_count
        counts += len(label_points)

    # Calculate the average localization error and matching accuracy
    if counts > 0:
        matching_accuracy = matched_counts / counts
        all_localization_error = all_localization_error / counts
    else:
        matching_accuracy = 0
        all_localization_error = 9999

    if matched_counts > 0:
        localization_error = matched_localization_error / matched_counts
    else:
        localization_error = 9999

    logging.info(f"Matched Localization Error: {localization_error:.4f}")
    logging.info(f"All Localization Error: {all_localization_error:.4f}")
    logging.info(f"Matching Accuracy: {matching_accuracy:.4f}")
    return {
        'localization_error': localization_error,
        'matching_accuracy': matching_accuracy
    }