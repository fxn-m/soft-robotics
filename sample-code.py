# -*- coding:utf-8 -*-
# Author: Qiukai Qi
# Date:


import numpy as np
import cv2 as cv


def sorting(centers_initial, centers_present):

    '''this function sorts the centers detected.
    '''
    centers_intermediate = np.ones((5, 2))
    # looping
    for i in range(5):
        for j in range(5):
            # calculating distance and judge
            if np.sqrt(np.sum(np.square(centers_initial[i]-centers_present[j]))) < 40:
                centers_intermediate[i] = centers_present[j]
                break
    centers_intermediate = centers_intermediate.astype(np.int16)
    return centers_intermediate


if __name__ == '__main__':

    file_path = 'output.avi'
    vc = cv.VideoCapture(file_path)

    if (vc.isOpened()== False):
        print("Error opening video stream or file")

    origin = np.array([80, 135])
    shape = np.array([320, 250])

    centers_temp = np.array([])
    while (vc.isOpened()):
        ret, img_rgb = vc.read()

        # cropping
        img_rgb = img_rgb[origin[0]:origin[0] + shape[0], origin[1]:origin[1] + shape[1]]
        # blurring
        img_rgb = cv.medianBlur(img_rgb, 5)

        img_hsv = cv.cvtColor(img_rgb, cv.COLOR_BGR2HSV)

        # detecting points by color
        red_hsv_lower = np.array([0, 50, 50])
        red_hsv_higher = np.array([10, 255, 255])
        mask1 = cv.inRange(img_hsv, lowerb=red_hsv_lower, upperb=red_hsv_higher)

        red_hsv_lower = np.array([156, 50, 50])
        red_hsv_higher = np.array([180, 255, 255])
        mask2 = cv.inRange(img_hsv, lowerb=red_hsv_lower, upperb=red_hsv_higher)
        mask = mask1 + mask2

        # detecting contours
        contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # finding centers of contours
        centers = []
        for cnt in contours:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            center = (int(x), int(y))

            cv.circle(img_rgb, center, 5, (255, 0, 0), -1)
            centers.append([center[0], center[1]])

        centers = np.array(centers)
        if len(centers_temp)==0:
            centers_temp = centers

        # sorting
        centers = sorting(centers_temp, centers)
        centers_temp = centers

        # drawing lines
        for i in range(len(centers)-1):
            cv.line(img_rgb, (centers[i, 0], centers[i, 1]), (centers[i+1, 0], centers[i+1, 1]), (255, 0, 0), 2)


        if ret:
            cv.imshow('image', img_rgb)

            if cv.waitKey(27) & 0xFF == ord('q'):
                break

        else:
            break

    cv.destroyAllWindows()