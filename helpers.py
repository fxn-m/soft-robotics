import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import os
import pandas as pd

# return the list of paths of video files in /recordings
def get_video_paths():
    video_paths = []
    for root, dirs, files in os.walk("recordings"):
        for file in files:
            if file.endswith(".MOV"):
                video_paths.append(os.path.join(root, file))
    return video_paths

def sorting(centers_initial, centers_present, n_markers):
    centers_intermediate = np.ones((n_markers, 2))
    for i in range(n_markers):
        for j in range(n_markers):
            if np.sqrt(np.sum(np.square(centers_initial[i]-centers_present[j]))) < 40:
                centers_intermediate[i] = centers_present[j]
                break
    centers_intermediate = centers_intermediate.astype(np.int16)
    return centers_intermediate

def angle_between_lines(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])

    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(cosine_angle)

    return angle * 180 / math.pi