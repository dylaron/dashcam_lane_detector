'''
My code is inspired by a YouTube Video
titled 'OpenCV Python Tutorial - Find Lanes for Self-Driving Cars (Computer Vision Basics Tutorial)'
by 'ProgrammingKnowledge'
https://www.youtube.com/watch?v=eLTLtUVuuy4

A simple demo video of my code below is available on my Instagram
https://www.instagram.com/p/B3RBbJrHNyq/?igshid=YmMyMTA2M2Y=
'''

import cv2
import numpy as np


def apply_graycanny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    else:
        cv2.putText(line_image,'Lane not detected',(10,10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)
    return line_image


def make_cordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = int(image.shape[0] * 0.82)
    y2 = int(image.shape[0] * 0.56)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, xlines):
    left_fit = []
    right_fit = []
    if xlines is not None:
        for line in xlines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if abs(slope) > 0.4:
                if slope < 0:
                    left_fit.append((slope, intercept))
                else:
                    right_fit.append((slope, intercept))
    if left_fit.__len__() > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_cordinates(image, left_fit_average)
    else:
        left_line = [300,576,301,575]
    if right_fit.__len__() > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_cordinates(image, right_fit_average)
    else:
        right_line = [911,576,910,575]
    return np.array([left_line, right_line])


def to_mask_img(image):
    mask = np.zeros(image.shape, dtype=np.uint8)
    roi_top = mask.shape[0] * .52
    roi_btm = mask.shape[0] * .80
    roi_top_ctr = 590
    roi_wid_bot_half = 420
    roi_wid_top_half = 20
    triangle = np.array([[(roi_top_ctr - roi_wid_bot_half, roi_btm), (roi_top_ctr + roi_wid_bot_half, roi_btm),
                          (roi_top_ctr + roi_wid_top_half, roi_top), (roi_top_ctr - roi_wid_top_half, roi_top)]], dtype=np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# frame = cv2.imread('dashcam_still.png')
cap = cv2.VideoCapture("F:/Videos/dashcam/2019/09/2019_0910_164059_099.MOV")

while cap.isOpened():
    ret, frame = cap.read()
    canny = apply_graycanny(frame)
    masked = to_mask_img(canny)
    H_lines = cv2.HoughLinesP(masked, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=40)
    average_lines = average_slope_intercept(frame, H_lines)
    line_image = display_lines(frame, average_lines)
    combo_image = cv2.addWeighted(frame, 0.7, line_image, 1, 1)
    #final_image = cv2.addWeighted(combo_image, 0.9, cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR), 0.2, 1)
    cv2.imshow('Lane detection by DyLaron', combo_image)
    # cv2.imshow('Mask', masked)
    keyinput = cv2.waitKey(3) & 0xFF
    if keyinput == 27:
        break
    elif keyinput == ord(' '):
        cv2.waitKey(0)
    elif keyinput == 100:
        for i in range(0, 60):
            _, frame = cap.read()

cv2.destroyAllWindows()
cap.release()
