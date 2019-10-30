import numpy as np
import cv2
import random
from bars.code import rectangle_utils, manipulate_image


def check_for_templates(img, bar, templates):

    for template in templates:

        # Store width and heigth of template in w and h
        # w, h = template.shape[::-1]

        _temp = template
        _bar = get_bar_mat(img, bar)

        # Convert it to grayscale
        # _img = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)
        # _temp = cv2.cvtColor(_temp, cv2.COLOR_BGR2GRAY)

        # Perform match operations.
        # res = cv2.matchTemplate(_img.astype(np.float32), _temp.astype(np.float32), cv2.TM_SQDIFF)
        # res = cv2.matchTemplate(_img.astype(np.float32), _temp.astype(np.float32), cv2.TM_SQDIFF_NORMED)
        # res = cv2.matchTemplate(_img.astype(np.float32), _temp.astype(np.float32), cv2.TM_CCORR)

        res = cv2.matchTemplate(_bar.astype(np.float32), _temp.astype(np.float32), cv2.TM_CCORR_NORMED)

        # res = cv2.matchTemplate(_img.astype(np.float32), _temp.astype(np.float32), cv2.TM_CCOEFF)
        # res = cv2.matchTemplate(_img.astype(np.float32), _temp.astype(np.float32), cv2.TM_CCOEFF_NORMED)

        # Specify a threshold
        # threshold = 0.935
        threshold = 0.97

        # Store the coordinates of matched area in a numpy array
        loc = np.where(res >= threshold)

        # Draw a rectangle around the matched region.
        for pt in zip(*loc[::-1]):
            # print(loc)
            # cv2.rectangle(img, pt, pt, (0, 255, 255), 10)

            # image_show.show_horizontal("matching", img, _bar, _temp)
            temp_img = np.copy(img)
            # rectangle_utils.draw_bar(temp_img, bar)
            # cv2.imshow("temp_img", temp_img)
            cv2.imshow("_bar", _bar)
            cv2.imshow("_template", _temp)
            # cv2.waitKey()

            return template

    return None


def get_slice_of_bar(img, bar, slice_size):
    w_slice = bar.get_width() * slice_size / 2
    h_slice = bar.get_height() * slice_size / 2

    smallest = min(w_slice, h_slice)

    x1 = int(rectangle_utils.get_center(bar)[0] - w_slice)
    x2 = int(rectangle_utils.get_center(bar)[0] + w_slice)
    y1 = int(rectangle_utils.get_center(bar)[1] - h_slice)
    y2 = int(rectangle_utils.get_center(bar)[1] + h_slice)

    x1 = int(rectangle_utils.get_center(bar)[0] - smallest)
    x2 = int(rectangle_utils.get_center(bar)[0] + smallest)
    y1 = int(rectangle_utils.get_center(bar)[1] - smallest)
    y2 = int(rectangle_utils.get_center(bar)[1] + smallest)

    bar_slice = img[y1:y2, x1:x2]

    return bar_slice


def get_bar_mat(img, bar):
    bar_slice = img[bar.get_y():bar.get_y() + bar.get_height(),
                    bar.get_x():bar.get_x() + bar.get_width()]

    return bar_slice


def rand_color():
    return random.randrange(0,255),random.randrange(0,255),random.randrange(0,255)


def find_classes(img, bars, draw):
    bar_classes = []
    templates = []
    tc = []
    # slice_size = 0.8
    slice_size = 0.4

    for bar in bars:
        bar_class = check_for_templates(img, bar, templates)

        if bar_class is None:
            bar_class = get_slice_of_bar(img, bar, slice_size)
            templates.append(bar_class)
            tc.append((bar_class, rand_color()))
            _bar = get_bar_mat(img, bar)
            cv2.imshow("new_template", _bar)

        for t,c in tc:
            if t is bar_class:
                rectangle_utils.draw_bar(draw, bar, c)

        cv2.imshow("temp_img", draw)
        cv2.waitKey()
        bar_classes.append((bar, bar_class))

    return bar_classes


def scan_chart(img):
    bars, legend = manipulate_image.find_features(np.copy(img))

    bars = rectangle_utils.filter_by_width2(bars, img, m=5)

    # for bar in bars:
    #     temp_img = np.copy(img)
    #     rectangle_utils.draw_bar(temp_img, bar)
    #     cv2.imshow("imss", temp_img)
    #     cv2.waitKey()

    bar_classes = find_classes(np.copy(img), bars, np.copy(img))

    for bar_class in bar_classes:
        bar = bar_class[0]
        _class = bar_class[1]

        # image_show.show_horizontal("bc", _class)
        # cv2.waitKey()

    # all_text, x_axis_text, y_axis_text, bar_text, legend_text, title_text = \
    #     text_finder.find_text_features(np.copy(img), bars, legend, epsilon=5)
    #
    # allthings= \
    #     text_finder.find_text_features(np.copy(img), bars, legend, epsilon=5)
    #
    # text_finder.draw_text(img, all_text)
    # image_show.show_horizontal("text", img)
    # cv2.waitKey()


"""

def draw_vertical_extremes(cont, img):
    extTop, extBot = get_vertical_extremes(cont)
    cv2.circle(img, extTop, 3, (0, 0, 255), -1)
    cv2.circle(img, extBot, 3, (255, 0, 0), -1)


def get_vertical_extremes(cont):
    extTop = tuple(cont[cont[:, :, 1].argmin()][0])
    extBot = tuple(cont[cont[:, :, 1].argmax()][0])
    return extTop, extBot


def get_horizontal_extremes(cont):
    extLeft = tuple(cont[cont[:, :, 0].argmin()][0])
    extRight = tuple(cont[cont[:, :, 0].argmax()][0])
    return extLeft, extRight


def get_contour_width(cont):
    extLeft, extRight = get_horizontal_extremes(cont)
    return extRight[0] - extLeft[0]


def get_contour_height(cont):
    extTop, extBot = get_vertical_extremes(cont)
    return extBot[1] - extTop[1]

def get_contour_center(cont):
    M = cv2.moments(cont)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


orig = cv2.imread("../../res/imgs/1_PIER.tiff")
orig = orig[0:355, 0:]

# image = cv2.imread("../../res/imgs/3_PIER.tiff")

# image = cv2.imread("../../res/imgs/2_EXCITE.tiff")
# image = image[0:500, 0:]

# image = cv2.imread("../../res/imgs/3_EXCITE.tiff")

# rotate image
rotation_offset = random.randrange(6, 8)
rotation_offset = rotation_offset * random.randrange(-1, 1)
# rotation_offset = 6
print("rot offset", rotation_offset)
rotated = manipulate_image.rotate_image(orig, rotation_offset)

x_axis, y_axis = manipulate_image.find_axes(rotated, position_restriction=True)
rotation_fixed = manipulate_image.fix_image_rotation(rotated)

image = np.copy(rotation_fixed)
# image_show.show_vertical("rot",rotated,image)
# cv2.waitKey(0)

# copy the original image for drawing on
copy_for_draw = np.copy(image)

# find the bars of the bar chart
bars = manipulate_image.isolate_bars(image, correction=5)

# find how many units there are per pixel
unit_pixel = text_finder.get_unit_pixels(image, copy_for_draw)
unit_pixel = unit_pixel - 0.013 # 0.005



# calculate contours of the bars
contours, hierarchy = cv2.findContours(cv2.cvtColor(bars, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


# draw the contours on an image
for i in range(0, len(contours)):
    # only draw a contour if the area is above a threshold
    if cv2.contourArea(contours[i]) > 550 and cv2.contourArea(contours[i]) < 100000:
        # print(cv2.contourArea(contours[i]))
        cv2.drawContours(copy_for_draw, contours, i, (255, 0, 0), thickness=1)
        draw_vertical_extremes(contours[i], image)
        # print("height = ", get_contour_height(contours[i]))
        # print("width = ", get_contour_width(contours[i]))
        height = (get_contour_height(contours[i])) * unit_pixel
        height = round(height, 1)
        center = get_contour_center(contours[i])
        cv2.putText(image, str(height), (center[0]-35, center[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)



# image_show.show_vertical("raw - threshold - contours", rotated)


cv2.line(rotated, x_axis[0], x_axis[1], (255, 0, 0), 3)
cv2.line(rotated, y_axis[0], y_axis[1], (255, 0, 0), 3)

cv2.putText(rotated, "1", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(rotation_fixed, "2", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(bars, "3", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.putText(image, "4", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

image_show.show_vertical("raw - threshold - contours", rotated, rotation_fixed, bars)
cv2.waitKey(0)

image_show.show_vertical("raw - threshold - contours", rotated, image)
cv2.waitKey(0)

print("[INFO] finished.")


"""