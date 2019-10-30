import cv2
from src.graphs import manipulate_image
import numpy as np


def check_is_bar_graph(img):
    copy = np.copy(img)
    bars = manipulate_image.isolate_bars(copy)
    contours, hierarchy = cv2.findContours(cv2.cvtColor(bars, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    counter = 0
    # draw the contours on an image
    for i in range(0, len(contours)):
        # only draw a contour if the area is above a threshold
        if cv2.contourArea(contours[i]) > 550 and cv2.contourArea(contours[i]) < 100000:
            counter = counter + 1

    if counter > 2:
        return True
    else:
        return False
