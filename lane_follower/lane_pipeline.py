import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # or 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    # Group Lines into L&R

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1 + 1)  # <-- Calculating the slope.

            if math.fabs(slope) < 0.5:  # <-- Only consider extreme slope
                continue

            if slope <= 0:  # <-- If the slope is negative, left group.
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:  # <-- Otherwise, right group.
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])

    # Linear Representation of each Line Group

    min_y = int(img.shape[0] * (3 / 5))
    max_y = int(img.shape[0])

    left_pnv = []
    right_pnv = []

    if len(left_line_x) > 0 and len(left_line_y) > 0:
        poly_left = np.poly1d(np.polyfit(left_line_y, left_line_x, deg=1))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        left_pnv = [(left_x_start, max_y), (left_x_end, min_y)]

        cv2.line(img, (left_x_start, max_y), (left_x_end, min_y), color, thickness)

    if len(right_line_x) > 0 and len(right_line_y) > 0:
        poly_right = np.poly1d(np.polyfit(right_line_y, right_line_x, deg=1))

        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))

        right_pnv = [(right_x_start, max_y), (right_x_end, min_y)]

        cv2.line(img, (right_x_start, max_y), (right_x_end, min_y), color, thickness)

    return left_pnv, right_pnv


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(
        img,
        rho,
        theta,
        threshold,
        np.array([]),
        minLineLength=min_line_len,
        maxLineGap=max_line_gap,
    )
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    left_pnv, right_pnv = draw_lines(line_img, lines)
    return line_img, left_pnv, right_pnv


def weighted_img(img, initial_img, alpha=0.8, beta=1.0, gamma=0.0):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * alpha + img * beta + gamma
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def crop_roi(image, im_channel_first=True):
    if im_channel_first:
        # moves channels to last dimension
        image = np.moveaxis(image, 0, 2)

    height, width = image.shape[:2]

    region_of_interest_vertices = [
        (0, height),
        (0, height / 1.5),
        (width / 2, height / 2),
        (width, height / 1.5),
        (width, height),
    ]

    gray_image = grayscale(image)
    # gray_image = gaussian_blur(gray_image, kernel_size=5)
    # gray_image = canny(gray_image, low_threshold=50, high_threshold=150)
    gray_image = region_of_interest(
        gray_image,
        np.array([region_of_interest_vertices], np.int32),
    )

    return gray_image


def lane_pipeline(image, im_format="rgb", im_channel_first=True):
    if im_channel_first:
        # moves channels to last dimension
        image = np.moveaxis(image, 0, 2)

    height, width = image.shape[:2]

    region_of_interest_vertices = [
        (0, height),
        (0, height / 1.5),
        (width / 2, height / 2),
        (width, height / 1.5),
        (width, height),
    ]

    gray_image = grayscale(image)
    gray_image = gaussian_blur(gray_image, kernel_size=5)
    cannyed_image = canny(gray_image, low_threshold=50, high_threshold=150)
    cropped_image = region_of_interest(
        cannyed_image,
        np.array([region_of_interest_vertices], np.int32),
    )

    line_image, left_pnv, right_pnv = hough_lines(
        cropped_image,
        rho=6,
        theta=np.pi / 60,
        threshold=100,
        min_line_len=5,
        max_line_gap=55,
    )

    if len(left_pnv) == 0 and len(right_pnv) == 0:
        line_image, left_pnv, right_pnv = hough_lines(
            cropped_image,
            rho=3,
            theta=np.pi / 10,
            threshold=35,
            min_line_len=2,
            max_line_gap=35,
        )

    final_img = weighted_img(line_image, image, alpha=0.8, beta=1.0)

    return final_img, left_pnv, right_pnv
