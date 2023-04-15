import numpy as np
import cv2 as cv
import pandas as pd

from helpers import * 

def extract_angles(file_path):
    vc = cv.VideoCapture(file_path)

    if (vc.isOpened() == False):
        print("Error opening video stream or file")

    # Get the total number of frames in the video
    # total_frames = int(vc.get(cv.CAP_PROP_FRAME_COUNT))
    # Set the current position to the last frame
    # vc.set(cv.CAP_PROP_POS_FRAMES, total_frames - 1)

    centers_temp = np.array([])
    n_markers = 7

    angles_list = []

    while (vc.isOpened()):
        ret, img_rgb = vc.read()

        if not ret or img_rgb is None:
            break

        img_rgb = cv.medianBlur(img_rgb, 5)
        img_rgb = cv.rotate(img_rgb, cv.ROTATE_90_CLOCKWISE)

        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)

        red_hsv_lower = np.array([0, 50, 50])
        red_hsv_higher = np.array([10, 255, 255])
        mask1 = cv.inRange(img_hsv, lowerb=red_hsv_lower,
                           upperb=red_hsv_higher)

        red_hsv_lower = np.array([156, 50, 50])
        red_hsv_higher = np.array([180, 255, 255])
        mask2 = cv.inRange(img_hsv, lowerb=red_hsv_lower,
                           upperb=red_hsv_higher)
        mask = mask1 + mask2

        contours, hierarchy = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        centers = []
        for cnt in contours:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            if radius > 5:
                center = (int(x), int(y))

                cv.circle(img_rgb, center, 5, (255, 0, 0), -1)
                centers.append([center[0], center[1]])

        centers = np.array(centers)
        if len(centers_temp) == 0:
            centers_temp = centers

        centers = sorting(centers_temp, centers, n_markers)
        centers_temp = centers

        # calculating angles
        angles = []
        for i in range(len(centers)-2):
            angles.append(angle_between_lines(
                centers[i], centers[i+1], centers[i+2]))
            angles = [new_angle if new_angle > 90 else 180 - new_angle for new_angle in angles]
        angles_list.append(angles)
        
        # drawing lines
        for i in range(len(centers)-1):
            cv.line(img_rgb, (centers[i, 0], centers[i, 1]),
                    (centers[i+1, 0], centers[i+1, 1]), (255, 0, 0), 2)
            
        if ret:
            cv.imshow('image', cv.rotate(
                img_rgb, cv.ROTATE_90_COUNTERCLOCKWISE))
            
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()

    return angles_list[-1]


if __name__ == '__main__':
    results = {}
    clips = get_video_paths()

    for clip in clips:
        try:
            angles = extract_angles(clip)
            print(f"Angles at the last frame of {clip}:", angles)
        except IndexError:
            print(f"Could not extract angles for {clip}")
            angles = [0] * 5

        results[clip] = angles

    pd.DataFrame(results).to_csv("line_pair_angles.csv")
