import math
import cv2
from bars.code import rectangle_utils

from matplotlib.patches import Rectangle

ORIENTATION_HORIZONTAL = 0
ORIENTATION_VERTICAL = 1


class Text:

    def __init__(self, rect, txt, orientation):
        self.rect = rect
        self.txt = txt
        self.orientation = orientation

    def contains(self, other, x_thresh=0, y_thresh=0):
        return rectangle_utils.contains(self.rect, other.get_rect(), x_thresh, y_thresh)

    def get_center(self):
        return rectangle_utils.get_center(self.rect)

    def get_distance(self, other):
        return rectangle_utils.get_distance(self.rect, other.get_rect())

    def is_number(self):
        try:
            float(self.txt)
            return True
        except ValueError:
            return False

    def draw_on(self, img):
        cv2.putText(img, self.txt, (self.rect.get_x(), self.rect.get_y() + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)

    def get_text(self):
        return self.txt

    def get_orientation(self):
        return self.orientation

    def get_rect(self):
        return self.rect

    def get_x(self):
        return self.rect.get_x()

    def get_y(self):
        return self.rect.get_y()

    def get_width(self):
        return self.rect.get_width()

    def get_height(self):
        return self.rect.get_height()

    def set_text(self, txt):
        self.txt = txt

    def set_rect(self, rect):
        self.rect = rect

    def set_orientation(self, orientation):
        self.orientation = orientation
