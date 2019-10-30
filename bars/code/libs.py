# These are the libraries that i'm using in my code or plan on using

# Python(3.7.3)
# OpenCV(4.1.0)                                 pip3 install opencv-python
# Python Image Library -> Pillow(6.0.0)         pip3 install pillow
# Numpy(1.16.2)                                 pip3 install numpy
# Scipy(1.2.1)                                  pip3 install scipy
# Matplotlib(3.0.3)                             pip3 install matplotlib
# Imutils(0.5.2)                                pip3 install imutils
# Tesseract(4.0.0)                              pip3 install pytesseract
# Sklearn(0.21.3)                               pip3 install -U scikit-learn
# Utils(0.21.3)                               pip3 install utils

import cv2
import platform
import PIL.Image as Image
import numpy as np
import scipy
import matplotlib as mpl
import imutils
import pytesseract
import sklearn


print("You are running python", platform.python_version())
print("You are running opencv", cv2.__version__)
print("You are running pillow", Image.PILLOW_VERSION)
print("You are running numpy", np.version.version)
print("You are running scipy", scipy.version.version)
print("You are running matplotlib", mpl.__version__)
print("You are running imutils", imutils.__version__)
print("You are running tesseract", pytesseract.get_tesseract_version())
print("You are running sklearn", sklearn.__version__)
