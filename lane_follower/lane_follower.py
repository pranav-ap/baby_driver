import cv2
from PIL import Image
from .blue_utils import *


class LaneFollowerClassic:
    def __init__(self) -> None:
        open_path = (
            "./../camera_cal/calibration*.jpg"
        )
        save_path = "./../camera_cal/"
        chessboard_size = [9, 6]
        self.objpoints = []
        self.imgpoints = []
        self.objpoints, self.imgpoints = camera_calibrate(
            chessboard_size, open_path, save_path
        )

        calib3_img = mpimg.imread(
            r"./../camera_cal/calibration1.jpg"
        )
        calib3_und_img = undistort_img(calib3_img, self.objpoints, self.imgpoints)

    def predict(self, original_center):
        img = original_center.resize((1280, 720), Image.LANCZOS)
        img = np.array(img)
        und_img = undistort_img(img, self.objpoints, self.imgpoints)
        # img = und_img

        # Step 2. Create a thresholded binary image
        ksize = 5  # Sobel kernel size, choose a larger odd number to smooth gradient measurements

        # Apply each of the thresholding functions
        gradx = abs_sobel_thresh(img, orient="x", sobel_kernel=ksize, thresh=(10, 100))
        mag_binary = mag_thresh(img, sobel_kernel=ksize, mag_thresh=(20, 100))
        s_binary = s_select(img, thresh=(150, 255))
        combined = np.zeros_like(gradx)
        combined[(((gradx == 1) & (mag_binary == 1)) | (s_binary == 1))] = 1

        # Step 3. Perform a perspective transform
        persp_obj = perspective_trans(img)
        warped = persp_obj["warped"]
        dst = persp_obj["dst"]
        src = persp_obj["src"]
        Minv = persp_obj["Minv"]  # save for use in step 6

        pts1 = np.array(src, np.int32)
        pts1 = pts1.reshape((-1, 1, 2))
        cv2.polylines(img, [pts1], True, (255, 0, 0), 3)

        pts2 = np.array(dst, np.int32)
        pts2 = pts2.reshape((-1, 1, 2))
        cv2.polylines(warped, [pts2], True, (255, 0, 0), 3)

        pers_obj2 = perspective_trans(combined)
        binary_warped = pers_obj2["warped"]
        curv_obj = curvature_eval(binary_warped, nwindows=30, margin=40, minpix=40)
        left_fit = curv_obj["left_fit"]
        right_fit = curv_obj["right_fit"]
        offset = curv_obj["offset"]

        steer_left = curv_obj["left_fit"][0] < 0
        curvature = 0.5 * (curv_obj["left_curverad"] + curv_obj["right_curverad"])

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        im = map_color(img, Minv, binary_warped, img, left_fitx, right_fitx, ploty)

        return im, curvature, steer_left, offset
