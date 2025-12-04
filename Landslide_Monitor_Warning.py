import cv2
import numpy as np
import csv
import sys
import os


class HighPrecisionMonitor:
    def __init__(self, video_path):
        # 1. Initialize video reading
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")

        # Obtain raw video parameters
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video loaded successfully: {self.width}x{self.height} @ {self.fps}fps, Total frames: {self.total_frames}")

        # 2. Initialize output settings
        output_video_path = "Output_Video_Name"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
        self.video_writer = cv2.VideoWriter(output_video_path, fourcc, self.fps, (self.width * 2, self.height))

        # CSV output
        self.csv_file = open("Outputr_Data_Name", 'w', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Frame_ID", "Time_Sec", "ROI_Avg_Speed(px/frame)", "ROI_Max_Speed"])

        # 3. Read the first frame as the "absolute baseline".
        ret, self.ref_frame = self.cap.read()
        if not ret:
            raise ValueError("Video failed to load")
        self.ref_gray = cv2.cvtColor(self.ref_frame, cv2.COLOR_BGR2GRAY)

        # ==========================================
        # 4. Select ROI
        # ==========================================
        print("\n>>> Preparing the ROI selection window... <<<")

        MAX_DISPLAY_HEIGHT = 900

        # Calculate scaling ratio
        scale_ratio = 1.0
        if self.height > MAX_DISPLAY_HEIGHT:
            scale_ratio = MAX_DISPLAY_HEIGHT / self.height
            display_width = int(self.width * scale_ratio)
            display_height = int(self.height * scale_ratio)
            print(f"High-resolution video detected; temporarily scaled for display. (Scaling ratio: {scale_ratio:.4f})")

            display_frame = cv2.resize(self.ref_frame, (display_width, display_height))
        else:
            display_frame = self.ref_frame

        print(">>> Please follow these steps: Select the desired area in the pop-up window. Then press [SPACE] or [ENTER] to confirm. <<<")

        roi_small = cv2.selectROI("Select ROI", display_frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI")

        # Check if a valid region has been selected.
        if roi_small[2] == 0 or roi_small[3] == 0:
            raise ValueError("No valid ROI region selected!")

        # === Map coordinates back to the original resolution ===
        if scale_ratio != 1.0:
            self.roi_x = int(roi_small[0] / scale_ratio)
            self.roi_y = int(roi_small[1] / scale_ratio)
            self.roi_w = int(roi_small[2] / scale_ratio)
            self.roi_h = int(roi_small[3] / scale_ratio)
            print(
                f"Coordinate mapping complete: Display layer {roi_small} -> Original layer ({self.roi_x}, {self.roi_y}, {self.roi_w}, {self.roi_h})")
        else:
            self.roi_x, self.roi_y, self.roi_w, self.roi_h = roi_small

        # Boundary check
        self.roi_x = max(0, self.roi_x)
        self.roi_y = max(0, self.roi_y)
        self.roi_w = min(self.width - self.roi_x, self.roi_w)
        self.roi_h = min(self.height - self.roi_y, self.roi_h)

        # ==========================================
        # 5. Initialize high-precision feature detection (SIFT)
        # ==========================================
        print("Initializing the SIFT feature detector...")
        self.detector = cv2.SIFT_create()

        # Detecting feature points in a reference frame
        mask = np.ones_like(self.ref_gray, dtype=np.uint8) * 255
        mask[self.roi_y:self.roi_y + self.roi_h, self.roi_x:self.roi_x + self.roi_w] = 0  

        self.kp_ref, self.des_ref = self.detector.detectAndCompute(self.ref_gray, mask=mask)
        print(f"Number of feature points in the baseline frame (outside of ROI): {len(self.kp_ref)}")

        # Feature matcher (FLANN based matcher is faster for SIFT, but BF is safer)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # The stabilized image from the previous frame.
        self.prev_stabilized_gray = self.ref_gray.copy()

        # Visual base map
        self.hsv = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.hsv[..., 1] = 255

    def stabilize_frame(self, curr_frame_gray):
        """
        High-precision image stabilization using SIFT + RANSAC
        """
        # Detect current frame features
        kp_curr, des_curr = self.detector.detectAndCompute(curr_frame_gray, None)

        if des_curr is None or len(kp_curr) < 10:
            return curr_frame_gray  

        # Feature matching
        matches = self.matcher.knnMatch(self.des_ref, des_curr, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return curr_frame_gray

        # Extraction point pairs
        src_pts = np.float32([self.kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_curr[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Calculate the transformation matrix
        M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

        if M is None:
            return curr_frame_gray

        stabilized_frame = cv2.warpPerspective(curr_frame_gray, M, (self.width, self.height))
        return stabilized_frame

    def process(self):
        frame_idx = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_idx += 1
            print(f"Frame being processed: {frame_idx}/{self.total_frames}...", end='\r')

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 1. Image Stabilization
            stabilized_gray = self.stabilize_frame(curr_gray)

            # 2. Calculating dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_stabilized_gray,
                stabilized_gray,
                None,
                pyr_scale=0.5, levels=5, winsize=25, iterations=5, poly_n=7, poly_sigma=1.5, flags=0
            )

            # 3. Calculate speed and angle
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # 4. Extract ROI data
            roi_mag = mag[self.roi_y:self.roi_y + self.roi_h, self.roi_x:self.roi_x + self.roi_w]

            # Noise reduction
            valid_mag = roi_mag[roi_mag > 0.5]

            if len(valid_mag) > 0:
                avg_speed = np.mean(valid_mag)
                max_speed = np.max(valid_mag)
            else:
                avg_speed = 0
                max_speed = 0

            # 写入 CSV
            time_sec = frame_idx / self.fps
            self.csv_writer.writerow([frame_idx, f"{time_sec:.3f}", f"{avg_speed:.5f}", f"{max_speed:.5f}"])

            # 5. Generate visual video frames
            self.hsv[..., 0] = ang * 180 / np.pi / 2
            self.hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

            rgb_flow = cv2.cvtColor(self.hsv, cv2.COLOR_HSV2BGR)

            cv2.rectangle(rgb_flow, (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (0, 255, 0), 3)

            # Convert the stabilized original image back to color for stitching.
            stabilized_bgr = cv2.cvtColor(stabilized_gray, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(stabilized_bgr, (self.roi_x, self.roi_y),
                          (self.roi_x + self.roi_w, self.roi_y + self.roi_h), (0, 255, 0), 3)

            # Image stitching
            combined_frame = np.hstack((stabilized_bgr, rgb_flow))

            # Write to video
            self.video_writer.write(combined_frame)

            # Update the previous frame
            self.prev_stabilized_gray = stabilized_gray

        # Processing complete
        self.cap.release()
        self.video_writer.release()
        self.csv_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "Input_Video_Name"

    if not os.path.exists(video_path):
        print(f"Error: File not found {video_path}")
    else:
        try:
            monitor = HighPrecisionMonitor(video_path)
            monitor.process()
        except Exception as e:
            print(f"Error: {e}")