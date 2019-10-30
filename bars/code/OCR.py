import pytesseract
import cv2
import numpy as np
import re
from matplotlib.patches import Rectangle
from bars.code import text
from bars.code import manipulate_image


def recognize_text(image):
    h, w = image.shape[:2]
    orientation = text.ORIENTATION_HORIZONTAL

    if w * 2 < h:
        image = manipulate_image.rotate_image(image, 90)
        orientation = text.ORIENTATION_VERTICAL

    txt = pytesseract.image_to_string(image, config='-l eng --oem 1 --psm 6')
    return txt, orientation


def contours_to_rectangles(contours):
    rectangles = []

    # create a list of all contour bounding boxes
    for contour in contours:
        # text bounding box
        x, y, w, h = cv2.boundingRect(contour)
        rectangles.append(Rectangle((x, y), w, h))

    return rectangles


def locate_text(image):
    kernel_size = 5
    blur_factor = 3
    thresh = 254

    element_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    element_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    morph = cv2.morphologyEx(image, cv2.MORPH_OPEN, element_1, iterations=1)
    image = cv2.medianBlur(morph, blur_factor)
    morph = cv2.morphologyEx(image, cv2.MORPH_DILATE, element_2, iterations=1)
    morph = cv2.morphologyEx(morph, cv2.MORPH_ERODE, element_2, iterations=1)

    morph_blur = cv2.blur(morph, (blur_factor, blur_factor))
    morph_thresh = cv2.threshold(morph_blur, thresh, 255, cv2.THRESH_BINARY)[1]
    morph_thresh = cv2.bitwise_not(morph_thresh)

    contours, hierarchy = cv2.findContours(morph_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    text_bounds = contours_to_rectangles(contours)

    cv2.waitKey()

    return text_bounds


def find_text(image):
    copy = np.copy(image)

    # Convert image to gray scale
    gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

    # Remove bars from image
    no_bars = manipulate_image.remove_thick_lines(gray, correction=7)

    # detect text bounding boxes
    text_bounds = locate_text(no_bars)

    # Store text bounding boxes
    texts = []
    for rect in text_bounds:
        # crop bounding box from original image
        crop = no_bars[rect.get_y():rect.get_y() + rect.get_height(), rect.get_x():rect.get_x() + rect.get_width()]

        # recognize text (convert bounding box to string)
        txt, orientation = recognize_text(crop)

        # filter out non roman text
        txt = re.split("^a-zA-Z0-9_", txt)

        if txt.__len__() > 0:
            texts.append(text.Text(rect, txt, orientation))

    return texts
