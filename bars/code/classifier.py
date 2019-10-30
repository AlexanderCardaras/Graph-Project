import cv2
import numpy as np
import math
from sklearn.cluster import KMeans
from bars.code import image_show, visual, text_finder, manipulate_image


# TODO: create a smart get_cluster_count algorithm
def extract_data(img, texts):
    contours = extract_important_contours(img)
    colors = get_prominent_colors(img)
    legend = extract_legend(img, texts, contours, colors)
    data = extract_bar_data(img, texts, contours, colors, legend)

    # for l in legend:
    #     print(l)

    for d in data:
        print(d)


def get_closest_color(key, colors):
    lowest_dif = math.inf
    best_color = None
    for color in colors:
        dif = abs(color[0] - key[0]) + abs(color[1] - key[1]) + abs(color[2] - key[2])
        if dif < lowest_dif:
            lowest_dif = dif
            best_color = color

    return best_color, lowest_dif


def get_cluster_count(img):
    return 8


def get_prominent_colors(img, number_of_colors):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

    eroded = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    img = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, kernel)

    # reshape the image to be a list of pixels
    image = img.reshape((img.shape[0] * img.shape[1], 3))

    # cluster the pixel intensities
    clt = KMeans(n_clusters=number_of_colors)
    clt.fit(image)

    # grab the number of different clusters and create a histogram
    # based on the number of pixels assigned to each cluster
    num_labels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=num_labels)

    # normalize the histogram, such that it sums to one
    hist = hist.astype("float")
    hist /= hist.sum()

    colors = []

    for (percent, color) in zip(hist, clt.cluster_centers_):
        if percent > 0.01:
            colors.append(color)

    return colors


def extract_important_contours(img):
    # find the large shapes / bars of the bar chart
    bars = manipulate_image.get_bars(img, correction=5)
    bars = cv2.cvtColor(bars, cv2.COLOR_BGR2GRAY)
    contours, hierarchy = cv2.findContours(bars, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def extract_legend(img, texts, contours, colors):
    legend = []
    for contour in contours:
        # text bounding box
        x, y, w, h = cv2.boundingRect(contour)

        probing_point = (x+w + 50, y + (h/2))
        for text in texts:
            if text_finder.rectangle_contains_point(text.rect, probing_point):
                color, dif = get_closest_color(img[y + int(h / 2)][x + int(w / 2)], colors)
                legend.append((color, text.get_text()))
                break

    return legend


def extract_bar_data(img, texts, contours, colors, legend):
    data = []
    for contour in contours:
        # text bounding box
        x, y, w, h = cv2.boundingRect(contour)
        probing_point = (x + (w/2), y - 20)
        color, dif = get_closest_color(img[y + int(h / 2)][x + int(w / 2)], colors)
        bar_label = None
        bar_text = None

        for item in legend:
            if item[0] is color:
                bar_label = item[1]
                break

        for text in texts:
            if text_finder.rectangle_contains_point(text.rect, probing_point):
                bar_text = text.get_text()
                break

        if bar_label is not None and bar_text is not None:
            data.append((bar_label, bar_text))

    return data


def get_type(vis):
    copy = np.copy(vis.get_img())
    bars = manipulate_image.isolate_bars_by_color(copy, vis.get_prominent_colors(), 30)
    image_show.show_horizontal('m', bars)
    cv2.waitKey()


def get_orientation(vis):
    return visual.ORIENTATION_NONE


def get_grouping(img):

    return manipulate_image.find_corners()


def get_visual(img):
    vis = visual.Visual(img)
    vis.set_prominent_colors(get_prominent_colors(img))
    x = 0
    for c in vis.get_prominent_colors():
        cv2.rectangle(img, (x, 0), (x+25, 100), c, -1)
        image_show.show_horizontal("img", img)

        x+=25

    cv2.waitKey()
    image_show.show_horizontal("img", img)
    vis.set_type(get_type(vis))
    vis.set_orientation(get_orientation(vis))
    vis.set_grouping(get_grouping(img))
    return vis


