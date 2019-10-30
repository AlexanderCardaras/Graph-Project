# USAGE
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_01.jpg
# python text_recognition.py --east frozen_east_text_detection.pb --image images/example_04.jpg --padding 0.05
# Source https://www.pyimagesearch.com/2018/09/17/opencv-ocr-and-text-recognition-with-tesseract/
# import the necessary packages

# TODO: Fix text concatenation order
# TODO: Fix combine_text efficiency
# TODO: Fix sort text by height efficiency
import numpy as np
import pytesseract
import cv2
import re
import random
from matplotlib.patches import Rectangle
from bars.code import image_show
from bars.code import manipulate_image
from bars.code import text
from bars.code import OCR
from bars.code import rectangle_utils

import operator


def test_1_2(data):
	result = []
	dp = data.split("\n")[1:]
	for d in dp:
		d = d.split("\t")
		if len(d) == 12:
			x = int(d[6])
			y = int(d[7])
			w = int(d[8])
			h = int(d[9])
			conf = float(d[10])
			txt = d[11]

			if conf > 80:
				box = Rectangle((x, y), w, h)
				text_obj = text.Text(box, txt, text.ORIENTATION_HORIZONTAL)
				result.append((text_obj, conf))

	return result


def test_3_4(all_data, new_data):
	for n_data in new_data:
		contained = False
		for o_data in all_data:
			if o_data[0].contains(n_data[0]) or n_data[0].contains(o_data[0]):

				if n_data[1] > o_data[1]:
					all_data.remove(o_data)
					all_data.append(n_data)

				contained = True
				break
		if contained is not True:
			all_data.append(n_data)

	return all_data


def sort_text_by_height(all_data):
	all_text_temp = []
	for data in all_data:
		all_text_temp.append((data[0], int(data[0].get_rect().get_y())))

	all_text_temp = sorted(all_text_temp, key=operator.itemgetter(1))

	all_text = []
	for data in all_text_temp:
		all_text.append(data[0])

	return all_text


def test_find_text(img):
	no_bar = manipulate_image.remove_thick_lines(img, correction=7)

	data_1 = test_1_2(pytesseract.image_to_data(no_bar, config='--psm 4'))
	data_2 = test_1_2(pytesseract.image_to_data(no_bar, config='--psm 6'))
	data_3 = test_1_2(pytesseract.image_to_data(no_bar, config='--psm 11'))
	data_4 = test_1_2(pytesseract.image_to_data(no_bar, config='--psm 12'))

	all_data = []

	all_data = test_3_4(all_data, data_1)
	all_data = test_3_4(all_data, data_2)
	all_data = test_3_4(all_data, data_3)
	all_data = test_3_4(all_data, data_4)


	# for text_data in all_data:
	# 	text = text_data[0]
	# 	conf = text_data[1]
	# 	txt = text.get_text()
	# 	box = text.get_rect()
	# 	cv2.putText(img, txt, (box.get_x(), box.get_y() + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

	return sort_text_by_height(all_data)


def stroke_width_transform(image):
	""" Source: http://www.math.tau.ac.il/~turkel/imagepapers/text_detection.pdf """
	copy = np.copy(image)
	gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)


def combine_text(texts, x_thresh=4, y_thresh=4):
	""" combines text that is close together """
	new_text = []
	groups = []

	for txt1 in texts:
		new_group = []
		for txt2 in texts:
			if new_group.__contains__(txt2):
				continue

			if txt1.contains(txt2, x_thresh, y_thresh) or txt2.contains(txt1, x_thresh, y_thresh):
				for group in groups:
					if group.__contains__(txt1) or group.__contains__(txt2):
						for txt in new_group:
							if not group.__contains__(txt):
								group.append(txt)
						new_group = group

				if not new_group.__contains__(txt1):
					new_group.append(txt1)

				if not new_group.__contains__(txt2):
					new_group.append(txt2)

		if not groups.__contains__(new_group):
			groups.append(new_group)

	for group in groups:
		if len(group) < 2:
			new_text.append(group[0])
			continue

		txt1 = group[0]
		txt2 = group[1]
		new_x = min(txt1[0].get_x(), txt2[0].get_x())
		new_y = min(txt1[0].get_y(), txt2[0].get_y())
		new_x2 = max(txt1[0].get_x() + txt1[0].get_width(), txt2[0].get_x() + txt2[0].get_width())
		new_y2 = max(txt1[0].get_y() + txt1[0].get_height(), txt2[0].get_y() + txt2[0].get_height())
		new_txt = ""

		for txt in group:
			if txt[0].get_x() < new_x:
				new_txt = txt[1] + new_txt
			else:
				new_txt = new_txt + txt[1]

			new_x = min(txt[0].get_x(), new_x)
			new_y = min(txt[0].get_y(), new_y)

			new_x2 = max(txt[0].get_x() + txt[0].get_width(), new_x2)
			new_y2 = max(txt[0].get_y() + txt[0].get_height(), new_y2)

		new_w = new_x2 - new_x
		new_h = new_y2 - new_y

		new_rect = Rectangle((new_x, new_y), new_w, new_h)
		new_text.append((new_rect, new_txt))

	return new_text


def combine_rectangles(r1, r2):
	new_x1 = min(r1.get_x(), r2.get_x())
	new_y1 = min(r1.get_y(), r2.get_y())

	new_x2 = max(r1.get_x() + r1.get_width(), r2.get_x() + r2.get_width())
	new_y2 = max(r1.get_y() + r1.get_height(), r2.get_y() + r2.get_height())

	new_rect = Rectangle((new_x1, new_y1), new_x2 - new_x1, new_y2 - new_y1)
	return new_rect


def rectangle_contains_point(rect, point):
	if (rect.get_x() <= point[0]) and (rect.get_x() + rect.get_width() >= point[0]):
		if (rect.get_y() <= point[1]) and (rect.get_y() + rect.get_height() >= point[1]):
			return True

	return False


def rectangles_are_close(r1, r2, x_thresh, y_thresh):
	thresh_rect = Rectangle((r1.get_x() - x_thresh, r1.get_y() - y_thresh),
							r1.get_width() + x_thresh,
							r1.get_height() + y_thresh)

	p1 = (r2.get_x(), r2.get_y())
	p2 = (r2.get_x() + r2.get_width(), r2.get_y())
	p3 = (r2.get_x(), r2.get_y() + r2.get_height())
	p4 = (r2.get_x() + r2.get_width(), r2.get_y() + r2.get_height())

	if rectangle_contains_point(thresh_rect, p1) or rectangle_contains_point(thresh_rect, p2) or \
		rectangle_contains_point(thresh_rect, p3) or rectangle_contains_point(thresh_rect, p4):
		return True

	return False


def combine_close_rectangles(rectangles):
	unique_rectangles = []

	for rect in rectangles:
		found_match = False
		for unique_rectangle in unique_rectangles:
			if rectangles_are_close(rect, unique_rectangle, 20, 10) or rectangles_are_close(unique_rectangle, rect, 20, 5):

				unique_rectangles.remove(unique_rectangle)
				unique_rectangles.append(combine_rectangles(unique_rectangle, rect))
				found_match = True
				break

		if found_match is False:
			unique_rectangles.append(rect)

	return unique_rectangles



def get_numbers(results, image=None):
	numbers = []
	# loop over the results

	# for ((startX, startY, endX, endY), text) in results:
	for (rect, text) in results:
		x1 = rect.get_x()
		x2 = rect.get_x() + rect.get_width()
		y1 = rect.get_y()
		y2 = rect.get_y() + rect.get_height()

		# display the text OCR'd by Tesseract
		# print("OCR TEXT")
		# print("========")
		# print("{}\n".format(text))

		# strip out non-ASCII text so we can draw the text on the image
		# using OpenCV, then draw the text and a bounding box surrounding
		# the text region of the input image
		text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
		if image is not None:
			cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

		try:
			text = re.sub(r'[^\w\s]', '', text) # remove all non-characters from string
			# print("{}\n".format(text))
			if image is not None:
				draw_number(x1, y1, text, image)
				# draw_number(endX, startY, text, image)
				# cv2.putText(image, text, (startX, startY - 0), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 3)

			text = float(text)
			# numbers.append((int((startX+endX)/2), int((startY+endY)/2), text))
			new_rect = Rectangle((x1, y1), x2 - x1, y2-y1)
			numbers.append((new_rect, text))
		except ValueError:
			text = None

	return numbers


def draw_number(x, y, num, image):
		cv2.circle(image, (x, y), 3, (255, 0, 255), -1)


def draw_numbers(numbers, image):
	for (x, y, num) in numbers:
		cv2.circle(image, (x, y), 3, (255, 0, 0), -1)
		cv2.putText(image, str(num), (x-40, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


# def decode_predictions(scores, geometry, args):
# 	# grab the number of rows and columns from the scores volume, then
# 	# initialize our set of bounding box rectangles and corresponding
# 	# confidence scores
# 	(numRows, numCols) = scores.shape[2:4]
# 	rects = []
# 	confidences = []
#
# 	# loop over the number of rows
# 	for y in range(0, numRows):
# 		# extract the scores (probabilities), followed by the
# 		# geometrical data used to derive potential bounding box
# 		# coordinates that surround text
# 		scoresData = scores[0, 0, y]
# 		xData0 = geometry[0, 0, y]
# 		xData1 = geometry[0, 1, y]
# 		xData2 = geometry[0, 2, y]
# 		xData3 = geometry[0, 3, y]
# 		anglesData = geometry[0, 4, y]
#
# 		# loop over the number of columns
# 		for x in range(0, numCols):
# 			# if our score does not have sufficient probability,
# 			# ignore it
# 			if scoresData[x] < args["min_confidence"]:
# 				continue
#
# 			# compute the offset factor as our resulting feature
# 			# maps will be 4x smaller than the input image
# 			(offsetX, offsetY) = (x * 4.0, y * 4.0)
#
# 			# extract the rotation angle for the prediction and
# 			# then compute the sin and cosine
# 			angle = anglesData[x]
# 			cos = np.cos(angle)
# 			sin = np.sin(angle)
#
# 			# use the geometry volume to derive the width and height
# 			# of the bounding box
# 			h = xData0[x] + xData2[x]
# 			w = xData1[x] + xData3[x]
#
# 			# compute both the starting and ending (x, y)-coordinates
# 			# for the text prediction bounding box
# 			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
# 			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
# 			startX = int(endX - w)
# 			startY = int(endY - h)
#
# 			# add the bounding box coordinates and probability score
# 			# to our respective lists
# 			rects.append((startX, startY, endX, endY))
# 			confidences.append(scoresData[x])
#
# 	# return a tuple of the bounding boxes and associated confidences
# 	return rects, confidences


def find_col(numbers):
	best_col = []
	for (x1, y1, num1) in numbers:
		e = 10
		col = []
		for (x2, y2, num2) in numbers:
			if abs(x1 - x2) <= e:
				col.append((x2, y2, num2))

		if len(col) >= len(best_col):
			best_col = col

	return best_col


# def estimate_pixel_unit(num_gap, pix_dist, entries):
# 	total = 0
# 	most_common_num = np.bincount(num_gap).argmax()
# 	count = 0
# 	for e in entries:
# 		if e[0] == most_common_num:
# 			total = total + e[1]
# 			count = count + 1
#
# 	return most_common_num, total/count


# def get_num_pix_entries(col):
# 	num_gap = []
# 	pix_dist = []
# 	entries = []
#
# 	for i in range(1, len(col)):
# 		(x0, y0, num0) = col[i-1]
# 		(x1, y1, num1) = col[i]
#
# 		gap = abs(num0 - num1)
# 		dist = abs(y0-y1)
#
# 		num_gap.append(gap)
# 		pix_dist.append(dist)
# 		entries.append((gap, dist))
#
# 	return num_gap, pix_dist, entries


def draw_text(image, texts):
	for txt in texts:
		x = txt.get_x()
		y = txt.get_y()
		w = txt.get_width()
		h = txt.get_height()

		# draw text bounding box
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

		# draw text
		# cv2.putText(image, txt.get_text(), (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)


# def get_unit_pixels(image):
# 	print("[INFO] starting text detection...")
# 	results = find_text(image)
# 	numbers = get_numbers(results, draw_on_me)
#
# 	draw_text(draw_on_me, results)
# 	image_show.show_horizontal("", draw_on_me)
# 	cv2.waitKey(0)
#
# 	#clean up numbers
# 	numbers_clean = []
# 	for (rect, num) in numbers:
# 		x1 = rect.get_x()
# 		x2 = rect.get_x() + rect.get_width()
# 		y1 = rect.get_y()
# 		y2 = rect.get_y() + rect.get_height()
#
# 		x = int((x1+x2)/2)
# 		y = int((y1+y2)/2)
#
# 		numbers_clean.append((x, y, num))
#
# 	col = find_col(numbers_clean)
#
# 	draw_numbers(col, image)
#
# 	num_gap, pix_dist, entries = get_num_pix_entries(col)
# 	unit_pixels = estimate_pixel_unit(num_gap, pix_dist, entries)
# 	# print("every ", unit_pixels[1], " pixels is approximately ", unit_pixels[0], " units")
# 	print("1 pixel is approximately ", (unit_pixels[0]/unit_pixels[1]), " units")
# 	return unit_pixels[0]/unit_pixels[1]


def find_col(numbers):
	best_col = []
	for (x1, y1, num1) in numbers:
		e = 10
		col = []
		for (x2, y2, num2) in numbers:
			if abs(x1 - x2) <= e:
				col.append((x2, y2, num2))

		if len(col) >= len(best_col):
			best_col = col

	return best_col


def find_text_features(image, bars, legend, epsilon=0):
	all_text = OCR.find_text(image)
	vertical_groupings = rectangle_utils.get_vertical_groupings(all_text, rectangle_utils.ALIGNMENT_RIGHT, epsilon)
	horizontal_groupings = rectangle_utils.get_horizontal_grouping(all_text, rectangle_utils.ALIGNMENT_CENTER, epsilon)
	y_axis_text = vertical_groupings[0]
	x_axis_text = horizontal_groupings[0]
	bar_text_bottom = rectangle_utils.find_closest_match(bars, all_text, rectangle_utils.ALIGNMENT_BOTTOM,
																  rectangle_utils.ALIGNMENT_TOP)
	bar_text_top = rectangle_utils.find_closest_match(bars, all_text, rectangle_utils.ALIGNMENT_TOP,
																  rectangle_utils.ALIGNMENT_BOTTOM)

	legend_text_left = rectangle_utils.find_closest_match(legend, all_text, rectangle_utils.ALIGNMENT_LEFT,
														  rectangle_utils.ALIGNMENT_RIGHT)
	legend_text_right = rectangle_utils.find_closest_match(legend, all_text, rectangle_utils.ALIGNMENT_RIGHT,
														  rectangle_utils.ALIGNMENT_LEFT)
	title_text = None

	# if len(legend) == 0:
	#
	# else:


	for bar, txt in bar_text_top:
		temp_img = np.copy(image)

		cv2.rectangle(temp_img, (int(rectangle_utils.get_center(bar)[0]), int(rectangle_utils.get_center(bar)[1])),
					  (int(rectangle_utils.get_center(bar)[0]), int(rectangle_utils.get_center(bar)[1])),
					  (255, 0, 255), 5)

		cv2.rectangle(temp_img, (int(rectangle_utils.get_center(txt)[0]), int(rectangle_utils.get_center(txt)[1])),
					  (int(rectangle_utils.get_center(txt)[0]), int(rectangle_utils.get_center(txt)[1])),
					  (255, 0, 255), 5)

		image_show.show_horizontal("", temp_img)
		cv2.waitKey(0)

	return all_text, x_axis_text, y_axis_text, bar_text_top, bar_text_bottom, legend_text_left, legend_text_right, title_text

