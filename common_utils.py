import cv2
import numpy as np
from PIL import Image
import torch


def read_cv2_image(path, format="rgb", channel_first=True, resize=None, tor=False):
    im = cv2.imread(path)  # returns a numpy array in BGR format

    if resize is not None:
        # eg. resize=(640, 640)
        im = cv2.resize(im, resize, interpolation=cv2.INTER_AREA)

    if format == "rgb":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if channel_first:
        # move channels to first dimension - returns view
        im = np.moveaxis(im, 2, 0)

    if tor:
        im = torch.from_numpy(im)

    return im


def show_cv2_image(im, im_format="rgb", im_channel_first=True):
    im_out = np.copy(im)

    if im_format == "bgr":
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    if im_channel_first:
        # moves channels to last dimension - returns view
        im_out = np.moveaxis(im_out, 0, 2)

    # now convert to pil
    im_out = Image.fromarray(im_out)
    return im_out


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
	rect = np.zeros((4, 2), dtype = "float32")
	
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	
    # now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	
	# return the ordered coordinates
	return rect


def four_point_transform(image, pts):
	# obtain a consistent order of the points and unpack them
	# individually
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	# compute the width of the new image, which will be the
	# maximum distance between bottom-right and bottom-left
	# x-coordiates or the top-right and top-left x-coordinates
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	# compute the height of the new image, which will be the
	# maximum distance between the top-right and bottom-right
	# y-coordinates or the top-left and bottom-left y-coordinates
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	# now that we have the dimensions of the new image, construct
	# the set of destination points to obtain a "birds eye view",
	# (i.e. top-down view) of the image, again specifying points
	# in the top-left, top-right, bottom-right, and bottom-left
	# order
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	# return the warped image
	return warped


def draw_points(image_np, pts):
    for pt in pts:
        cv2.circle(image_np, tuple(pt), radius=3, color=(255, 255, 0), thickness=-1)

    return image_np


