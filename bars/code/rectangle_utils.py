import math
import statistics
import cv2
import numpy as np
from bars.code import list_utils

ALIGNMENT_CENTER = 0
ALIGNMENT_LEFT = 1
ALIGNMENT_RIGHT = 2
ALIGNMENT_TOP = 3
ALIGNMENT_BOTTOM = 4


def contains(rect1, rect2, x_thresh=0, y_thresh=0):
    r1_tl = (rect1.get_x() - x_thresh, rect1.get_y() - y_thresh)
    r1_br = (rect1.get_x() + rect1.get_width(), rect1.get_y() + rect1.get_height())

    r2_tl = (rect2.get_x(), rect2.get_y())
    r2_br = (rect2.get_x() + rect2.get_width() + x_thresh, rect2.get_y() + rect2.get_height() + y_thresh)

    return (r1_tl[0] < r2_br[0] and r1_br[0] > r2_tl[0]) and (r1_tl[1] < r2_br[1] and r1_br[1] > r2_tl[1])


def get_center(rect):
    x1 = rect.get_x()
    x2 = rect.get_x() + rect.get_width()
    y1 = rect.get_y()
    y2 = rect.get_y() + rect.get_height()

    return ((x1 + x2) / 2), ((y1 + y2) / 2)


def get_distance(rect1, rect2):

    x1, y1 = get_center(rect1)
    x2, y2 = get_center(rect2)

    return get_distance_points(x1, y1, x2, y2)


def get_distance_points(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1))


def get_vertical_groupings(all_text, alignment=ALIGNMENT_CENTER, epsilon=0):
    groups = []
    for text1 in all_text:
        group = [text1]
        for text2 in all_text:
            if text1 is not text2:
                if alignment == ALIGNMENT_CENTER:
                    x1, _ = text1.get_center()
                    x2, _ = text2.get_center()
                    if abs(x1-x2) < epsilon:
                        group.append(text2)
                elif alignment == ALIGNMENT_LEFT:
                    x1 = text1.get_x()
                    x2 = text2.get_x()
                    if abs(x1 - x2) < epsilon:
                        group.append(text2)
                elif alignment == ALIGNMENT_RIGHT:
                    x1 = text1.get_x() + text1.get_width()
                    x2 = text2.get_x() + text2.get_width()
                    if abs(x1 - x2) < epsilon:
                        group.append(text2)
                else:
                    print("Unsupported Alignment for get_vertical_groupings", alignment)
                    exit()

        groups.append(group)

    groups = list_utils.remove_overlapping_lists(groups)
    return groups


def get_horizontal_grouping(all_text, alignment=ALIGNMENT_CENTER, epsilon=0):
    groups = []
    for text1 in all_text:
        group = [text1]
        for text2 in all_text:
            if text1 is not text2:
                if alignment == ALIGNMENT_CENTER:
                    _, y1 = text1.get_center()
                    _, y2 = text2.get_center()
                    if abs(y1 - y2) < epsilon:
                        group.append(text2)
                elif alignment == ALIGNMENT_TOP:
                    y1 = text1.get_y()
                    y2 = text2.get_y()
                    if abs(y1 - y2) < epsilon:
                        group.append(text2)
                elif alignment == ALIGNMENT_BOTTOM:
                    y1 = text1.get_y() + text1.get_height()
                    y2 = text2.get_y() + text2.get_height()
                    if abs(y1 - y2) < epsilon:
                        group.append(text2)
                else:
                    print("Unsupported Alignment for get_horizontal_grouping", alignment)
                    exit()

        groups.append(group)

    groups = list_utils.remove_overlapping_lists(groups)
    return groups


def find_closest_match(list1, list2, l1_alignment=ALIGNMENT_CENTER, l2_alignment=ALIGNMENT_CENTER):
    best_matches = []
    for rect1 in list1:
        best_match = None
        distance = math.inf
        for rect2 in list2:
            if rect1 is not rect2:
                x1 = y1 = x2 = y2 = None
                if l1_alignment == ALIGNMENT_CENTER:
                    x1, y1 = get_center(rect1)
                elif l1_alignment == ALIGNMENT_LEFT:
                    x1, y1 = get_center(rect1)
                    x1 -= rect1.get_width()/2
                elif l1_alignment == ALIGNMENT_RIGHT:
                    x1, y1 = get_center(rect1)
                    x1 += rect1.get_width() / 2
                elif l1_alignment == ALIGNMENT_TOP:
                    x1, y1 = get_center(rect1)
                    y1 -= rect1.get_height()/2
                elif l1_alignment == ALIGNMENT_BOTTOM:
                    x1, y1 = get_center(rect1)
                    y1 += rect1.get_height() / 2
                else:
                    print("Unsupported Alignment for find_closest_vertical_match", l1_alignment)
                    exit()

                if l2_alignment == ALIGNMENT_CENTER:
                    x2, y2 = get_center(rect2)
                elif l2_alignment == ALIGNMENT_LEFT:
                    x2, y2 = get_center(rect2)
                    x2 -= rect2.get_width()/2
                elif l2_alignment == ALIGNMENT_RIGHT:
                    x2, y2 = get_center(rect2)
                    x2 += rect2.get_width() / 2
                elif l2_alignment == ALIGNMENT_TOP:
                    x2, y2 = get_center(rect2)
                    y2 -= rect2.get_height()/2
                elif l2_alignment == ALIGNMENT_BOTTOM:
                    x2, y2 = get_center(rect2)
                    y2 += rect2.get_height() / 2
                else:
                    print("Unsupported Alignment for find_closest_vertical_match", l1_alignment)
                    exit()

                temp_distance = get_distance_points(x1, y1, x2, y2)

                if temp_distance < distance:
                    distance = temp_distance
                    best_match = rect2

        best_matches.append((rect1, best_match))

    return best_matches


def filter_by_width(rects, m=1):
    temp = []
    for rect in rects:
        temp.append(abs(rect.get_width()))

    elements = np.array(temp)
    mean = np.mean(elements, axis=0)
    sd = np.std(elements, axis=0)

    final_list = []
    # print(mean,sd)
    for i in range(0, len(temp)):
        # print(temp[i])
        if temp[i] > mean - m * sd and temp[i] < mean + m * sd:
            final_list.append(rects[i])

    return final_list


def myround(x, base=5):
    return int(base * round(float(x)/base))


def filter_by_width2(rects, img, m=5):
    temp = []
    for rect in rects:
        temp.append(abs(rect.get_width()))

    temp = [myround(x, m) for x in temp]

    elements = np.array(temp)
    median = np.median(elements, axis=0)

    final_list = []

    # print(mean,sd)
    for i in range(0, len(temp)):
        # print(median, temp[i])

        if abs(median - temp[i]) <= m*2:
            # temp_img = np.copy(img)
            # draw_bar(temp_img, rects[i])
            # cv2.imshow("imss", temp_img)
            # cv2.waitKey()
            final_list.append(rects[i])

    return final_list


def draw_bar(img, bar, color=(0, 255, 255)):
    cv2.rectangle(img, (bar.get_x(),bar.get_y()), (bar.get_x() + bar.get_width(), bar.get_y() + bar.get_height()),
                  color, 5)
