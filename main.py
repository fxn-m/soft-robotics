# -*- coding:utf-8 -*-
# Author: Qiukai Qi & Felix Newport-Mangell
# Date: 15/04/2023

import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt

from helpers import * 

def update_plot(frame_count, angles_list):
    """
    Update the plot for each line pair
    Called for each frame by the animation function
    """
    for i, line in enumerate(plot_lines):
        line.set_data(range(frame_count), [
            angles[i] for angles in angles_list])
    ax.relim()
    ax.autoscale_view()
 
if __name__ == '__main__':

    file_path = f'./recordings/IMG_{3700}.mov'
    vc = cv.VideoCapture(file_path)

    if (vc.isOpened() == False):
        print("Error opening video stream or file")

    centers_temp = np.array([])
    n_markers = 7

    # Set up the matplotlib figure and axis
    fig, ax = plt.subplots()
    ax.set_title("Angle at each pivot")
    ax.set_xlabel("Frame")
    ax.set_ylabel("Angle (degrees)")

    # Initialize the plot lines for each line pair
    plot_lines = []
    for i in range(n_markers - 2):
        line, = ax.plot([], [], label=f"Pivot {i + 1}")
        plot_lines.append(line)
    ax.legend(loc=3)

    # Initialize the frame counter and angles_list
    frame_count = 0
    angles_list = []

    while (vc.isOpened()):
        ret, img_rgb = vc.read()

        # Check if the frame is empty and break the loop if it is
        if not ret or img_rgb is None:
            break

        # blurring
        img_rgb = cv.medianBlur(img_rgb, 5)
        img_rgb = cv.rotate(img_rgb, cv.ROTATE_90_CLOCKWISE)

        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)

        # detecting points by color
        red_hsv_lower = np.array([0, 50, 50])
        red_hsv_higher = np.array([10, 255, 255])
        mask1 = cv.inRange(img_hsv, lowerb=red_hsv_lower,
                           upperb=red_hsv_higher)

        red_hsv_lower = np.array([156, 50, 50])
        red_hsv_higher = np.array([180, 255, 255])
        mask2 = cv.inRange(img_hsv, lowerb=red_hsv_lower,
                           upperb=red_hsv_higher)
        mask = mask1 + mask2

        # detecting contours
        contours, hierarchy = cv.findContours(
            mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # finding centers of contours
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

        # sorting
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
            img_rgb = cv.flip(img_rgb, 0)
            cv.imshow('image', cv.rotate(
                img_rgb, cv.ROTATE_90_COUNTERCLOCKWISE))

            # Update the plot with the new angles
            frame_count += 1
            update_plot(frame_count, angles_list)
            plt.pause(0.001)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break

    cv.destroyAllWindows()

    # erase existing plot
    plt.clf()

    # After the video processing loop
    angles_array = np.array(angles_list)

    # Plot the angles for each adjacent line pair
    for i in range(angles_array.shape[1]):
        plt.plot(angles_array[:, i], label=f"Pivot {i+1}")

    plt.xlabel("Frame")
    plt.ylabel("Angle (degrees)")
    plt.title("Angle at each pivot")

    plt.legend()

    # Export the plot as a PNG
    plt.savefig('angles_per_line_pair.png', dpi=600)
    plt.show()
