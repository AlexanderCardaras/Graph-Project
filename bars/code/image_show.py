import cv2
import numpy as np

def make_three_dim(img):
    """ Returns a 3-channel copy of the image given, or the original image if it is already 3-channel """

    # Is the image 2-channel?
    if len(img.shape) == 2:
        blank_image = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        img2 = np.zeros_like(blank_image)
        img2[:, :, 0] = img
        img2[:, :, 1] = img
        img2[:, :, 2] = img
        return img2

    return img


def show_horizontal(name, *imgs):
    """ Displays all images horizontally next to each other """

    # Set window properties (fullscrean)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Create an empty array to store 3-channel images
    three_dim_images = []

    # Convert 2-channel(grayscale) images to 3-channel(bgr) images
    # We need all images to be 3-channel so we can concatenate them.
    for image in imgs:
        three_dim_images.append(make_three_dim(image))

    # Concatenate/Combine images horizontally
    horizontal = np.concatenate(three_dim_images, axis=1)
    cv2.imshow(name, horizontal)


def show_vertical(name, *imgs):
    """ Displays all images vertically next to each other """

    # Set window properties (fullscrean)
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # Create an empty array to store 3-channel images
    three_dim_images = []

    # Convert 2-channel(grayscale) images to 3-channel(bgr) images
    # We need all images to be 3-channel so we can concatenate them.
    for image in imgs:
        three_dim_images.append(make_three_dim(image))

    # Concatenate/Combine images vertically
    vertical = np.concatenate(three_dim_images, axis=0)
    cv2.imshow(name, vertical)


def show_rectangles(rects, img, color=(255, 0, 255)):
    for rect in rects:
        cv2.rectangle(img, rect.get_xy(), (rect.get_x() + rect.get_width(),
                                           rect.get_y() + rect.get_height()), color, -1)
    cv2.imshow("rects", img)
    cv2.waitKey()
