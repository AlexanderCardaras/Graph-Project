import math
from matplotlib.patches import Rectangle


TYPE_NONE = -1
TYPE_BAR = 0
TYPE_LINE = 1
TYPE_TABLE = 3

ORIENTATION_NONE = -1
ORIENTATION_HORIZONTAL = 0
ORIENTATION_VERTICAL = 1

GROUPING_NONE = -1
GROUPING_SEPARATE = 0
GROUPING_TOGETHER = 1


class Visual:

    def __init__(self, img):
        self.img = img
        self.prominent_colors = []

        self.type = TYPE_NONE
        self.orientation = ORIENTATION_NONE
        self.grouping = GROUPING_NONE

    def set_img(self, img):
        self.img = img

    def set_type(self, type):
        self.type = type

    def set_orientation(self, orientation):
        self.orientation = orientation

    def set_prominent_colors(self, prominent_colors):
        self.prominent_colors = prominent_colors

    def set_grouping(self, grouping):
        self.grouping = grouping

    def get_img(self):
        return self.img

    def get_type(self):
        return self.type

    def get_orientation(self):
        return self.orientation

    def get_prominent_colors(self):
        return self.prominent_colors

    def get_grouping(self):
        return self.grouping
