from bars.code import bar_chart
import numpy as np
import cv2

# orig = cv2.imread("../../res/imgs/1_PIER.tiff")
# orig = cv2.imread("../../res/imgs/column-chart.png")
# orig = cv2.imread("../../res/imgs/region.png")
# orig = cv2.imread("../../res/imgs/bar_graph_cross_hatch.png")
orig = cv2.imread("../../res/imgs/bar_hatch.png")
# orig = cv2.imread("../../res/imgs/bar_pat.png")

# orig = orig[355*0:360*1, 30:]
# orig = orig[355*0:355*3, 0:]
# orig = orig[355*1:355*2, 0:]
# orig = orig[355*2:355*3, 0:]

copy = np.copy(orig)

bar_chart.scan_chart(copy)
