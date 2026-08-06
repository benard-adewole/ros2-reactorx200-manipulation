#!/usr/bin/env python3

"""!
Class to represent the camera.
"""
 
import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

import cv2
import time
import numpy as np
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import *
from cv_bridge import CvBridge, CvBridgeError
from utils import apply_homography, world_to_frame, get_orientation, retrieve_area_color, retrieve_center_color, get_size, get_shape
from constant import font, colors, OFFSET_A, OFFSET_B, FACTORY_DEPTH_INTRINSIC_MATRIX, FACTORY_DEPTH_DISTCOEFFS, colors_hsv


class Camera():
    """!
    @brief      This class describes a camera.
    """

    def __init__(self):
        """!
        @brief      Construcfalsets a new instance.
        """
        self.VideoFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.GridFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.TagImageFrame = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRaw = np.zeros((720,1280)).astype(np.uint16)
        self.DepthFrameWarped = np.zeros((720,1280)).astype(np.uint16)
        """ Extra arrays for colormaping the depth image"""
        self.DepthFrameHSV = np.zeros((720,1280, 3)).astype(np.uint8)
        self.DepthFrameRGB = np.zeros((720,1280, 3)).astype(np.uint8)


        # mouse clicks & calibration variables
        self.camera_calibrated = False
        self.intrinsic_matrix = np.eye(3)
        self.extrinsic_matrix = np.eye(4)
        self.last_click = np.array([0, 0]) # This contains the last clicked position
        self.new_click = False # This is automatically set to True whenever a click is received. Set it to False yourself after processing a click
        self.rgb_click_points = np.zeros((5, 2), int)
        self.depth_click_points = np.zeros((5, 2), int)
        self.grid_x_points = np.arange(-450, 500, 50)
        self.grid_y_points = np.arange(-175, 525, 50)
        self.grid_points = np.array(np.meshgrid(self.grid_x_points, self.grid_y_points))
        self.tag_detections = np.array([])
        self.tag_locations = [[-250, -25], [250, -25], [250, 275], [-250, 275]]
        self.H = np.eye(3)
        """ block info """
        self.block_detected = False
        self.offset = np.ones((1280, 1), dtype=np.float32) * (np.arange(720) * OFFSET_A + OFFSET_B)
        self.block_contours = np.array([])
        self.block_detections = np.array([])

    def processVideoFrame(self):
        """!
        @brief      Process a video frame
        """
        cv2.drawContours(self.VideoFrame, self.block_contours, -1,
                         (255, 0, 255), 3)

    def ColorizeDepthFrame(self):
        """!
        @brief Converts frame to colormaped formats in HSV and RGB
        """
        self.DepthFrameHSV[..., 0] = self.DepthFrameRaw >> 1
        self.DepthFrameHSV[..., 1] = 0xFF
        self.DepthFrameHSV[..., 2] = 0x9F
        self.DepthFrameRGB = cv2.cvtColor(self.DepthFrameHSV,
                                          cv2.COLOR_HSV2RGB)

    def loadVideoFrame(self):
        """!
        @brief      Loads a video frame.
        """
        self.VideoFrame = cv2.cvtColor(
            cv2.imread("data/rgb_image.png", cv2.IMREAD_UNCHANGED),
            cv2.COLOR_BGR2RGB)
        

    def loadDepthFrame(self):
        """!
        @brief      Loads a depth frame.
        """
        self.DepthFrameRaw = cv2.imread("data/raw_depth.png",
                                        0).astype(np.uint16)

    def convertQtVideoFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.VideoFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtGridFrame(self):
        """!
        @brief      Converts frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.GridFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtDepthFrame(self):
        """!
       @brief      Converts colormaped depth frame to format suitable for Qt

       @return     QImage
       """
        try:
            img = QImage(self.DepthFrameRGB, self.DepthFrameRGB.shape[1],
                         self.DepthFrameRGB.shape[0], QImage.Format_RGB888)
            return img
        except:
            return None

    def convertQtTagImageFrame(self):
        """!
        @brief      Converts tag image frame to format suitable for Qt

        @return     QImage
        """

        try:
            frame = cv2.resize(self.TagImageFrame, (1280, 720))
            img = QImage(frame, frame.shape[1], frame.shape[0],
                         QImage.Format_RGB888)
            return img
        except:
            return None

    def getAffineTransform(self, coord1, coord2):
        """!
        @brief      Find the affine matrix transform between 2 sets of corresponding coordinates.

        @param      coord1  Points in coordinate frame 1
        @param      coord2  Points in coordinate frame 2

        @return     Affine transform between coordinates.
        """
        pts1 = coord1[0:3].astype(np.float32)
        pts2 = coord2[0:3].astype(np.float32)
        # print(cv2.getAffineTransform(pts1, pts2))
        return cv2.getAffineTransform(pts1, pts2)

    def loadCameraCalibration(self, file):
        """!
        @brief      Load camera intrinsic matrix from file.

                    TODO: use this to load in any calibration files you need to

        @param      file  The file
        """
        pass

    
    def transform_images(self):
        """
        Returns:
            tuple: Transformed RGB image, Transformed depth image.
        """
        # Load the RGB and depth images
        rgb_img = self.VideoFrame
        depth_img = self.DepthFrameRaw

        # Undistort depth using depth factory intrinsic matrix
        # depth_img = cv2.undistort(depth_img, FACTORY_DEPTH_INTRINSIC_MATRIX, FACTORY_DEPTH_DISTCOEFFS)

        # Use to ensure that the entire transformed image is contained within the output
        scale_factor = 1.0

        h, w = rgb_img.shape[:2]
        
        T_i = self.extrinsic_matrix
        K = self.intrinsic_matrix
        T_f = np.array([
            [1, 0,  0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 1000],
            [0, 0,  0, 1]
        ])

        # Calculate the relative transformation matrix between the initial and final camera poses
        T_relative = np.dot(T_f, np.linalg.inv(T_i))
        
        # Compute the new projection matrix for the RGB image
        P_new = np.dot(K, T_relative[:3, :3])
        P_inv = np.linalg.inv(P_new)
        
        # Compute the homography for RGB image
        H_rgb = self.H

        u = np.repeat(np.arange(w)[None, :], h, axis=0)
        v = np.repeat(np.arange(h)[:, None], w, axis=1)
        
        Z = depth_img
        X = (u - K[0,2]) * Z / K[0,0]
        Y = (v - K[1,2]) * Z / K[1,1]
        
        # Homogeneous coordinates in the camera frame
        points_camera_frame = np.stack((X, Y, Z, np.ones_like(Z)), axis=-1)
        
        # Apply the relative transformation to the depth points
        points_transformed = np.dot(points_camera_frame, T_relative.T)
        
        # Project back to depth values
        depth_transformed = points_transformed[..., 2]
        
        # Create a larger canvas for depth
        enlarged_h_depth, enlarged_w_depth = int(h * scale_factor), int(w * scale_factor)
        
        # Use the same homography as RGB for depth
        warped_depth = cv2.warpPerspective(depth_transformed, H_rgb, (enlarged_w_depth, enlarged_h_depth))
        # Apply manual homography offset
        # the source and dest points based off measuring block in frame and correcting center of block
        src_depth_offset = np.array([
            (380, 533),
            (285, 163),
            (988, 145),
            (989, 557)
        ], dtype=np.float32)
        dst_depth_offset = np.array([
            (369, 538),
            (269, 153),
            (992, 135),
            (994, 564)
        ], dtype=np.float32)
        H_depth_offset = cv2.findHomography(src_depth_offset, dst_depth_offset)[0]
        warped_depth = cv2.warpPerspective(warped_depth, H_depth_offset, (enlarged_w_depth, enlarged_h_depth))
        
        # enlarged_h_offset, enlarged_w_offset = int(self.offset.shape[0] * scale_factor), int(self.offset.shape[1] * scale_factor)
        # warped_offset = cv2.warpPerspective(self.offset, H_rgb)
        # warped_offset = cv2.resize(self.offset, (enlarged_h_depth, enlarged_w_depth))

        warped_depth -= self.offset.T
        self.DepthFrameWarped = warped_depth

        


    def blockDetector(self):
        """!
        @brief      Detect blocks from rgb

                    TODO: Implement your block detector here. You will need to locate blocks in 3D space and put their XYZ
                    locations in self.block_detections
        """
        pass
        rgb_image = self.VideoFrame
        cnt_image = rgb_image.copy()
        depth_image = self.DepthFrameWarped
        block_detections = np.array([])
        lower = 870
        upper = 978

        # mask out arm & outside board
        mask = np.zeros_like(depth_image, dtype=np.uint8)

        board_dim = [(120, 3), (1165, 684)]
        arm_dim = [(560, 375), (720, board_dim[1][1])]
        left_post_dim = [(board_dim[0][0], 160), (145, 480)]
        right_post_dim = [(1135, left_post_dim[0][1]), (board_dim[1][0], left_post_dim[1][1])]

        cv2.rectangle(mask, board_dim[0], board_dim[1], 255, cv2.FILLED)
        cv2.rectangle(mask, arm_dim[0], arm_dim[1], 0, cv2.FILLED)
        cv2.rectangle(mask, left_post_dim[0], left_post_dim[1], 0, cv2.FILLED)
        cv2.rectangle(mask, right_post_dim[0], right_post_dim[1], 0, cv2.FILLED)
        # cv2.rectangle(cnt_image, board_dim[0], board_dim[1], (255, 0, 0), 2)
        # cv2.rectangle(cnt_image, arm_dim[0], arm_dim[1], (0, 0, 255), 2)
        # cv2.rectangle(cnt_image, left_post_dim[0], left_post_dim[1], (0, 0, 255), 2)
        # cv2.rectangle(cnt_image, right_post_dim[0], right_post_dim[1], (0, 0, 255), 2)

        #Mask for Added april tags TODO: Get rid of these for competition, just for testing & simplicity
        april_tag5 = [(board_dim[0][0], 600), (250, board_dim[1][1])]
        april_tag6 = [(1050, 550), board_dim[1]]
        april_tag7 = [(1070, board_dim[0][1]), (board_dim[1][0], 120)]
        april_tag8 = [board_dim[0], (250, 150)]

        cv2.rectangle(mask, april_tag5[0], april_tag5[1], 0, cv2.FILLED)
        cv2.rectangle(mask, april_tag6[0], april_tag6[1], 0, cv2.FILLED)
        cv2.rectangle(mask, april_tag7[0], april_tag7[1], 0, cv2.FILLED)
        cv2.rectangle(mask, april_tag8[0], april_tag8[1], 0, cv2.FILLED)
        # cv2.rectangle(cnt_image, april_tag5[0], april_tag5[1], (0, 0, 255), 2)
        # cv2.rectangle(cnt_image, april_tag6[0], april_tag6[1], (0, 0, 255), 2)
        # cv2.rectangle(cnt_image, april_tag7[0], april_tag7[1], (0, 0, 255), 2)
        # cv2.rectangle(cnt_image, april_tag8[0], april_tag8[1], (0, 0, 255), 2)

        thresh = cv2.bitwise_and(cv2.inRange(depth_image, lower, upper), mask)

        # Clear the noise
        kernel_size = 4
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # clean the noise outside
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel) # fill the holes inside

        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # print("Number of contours found: ", len(contours))

        contours = [contour for contour in contours if cv2.contourArea(contour) > 500]
        # Get the depth of the blocks
        for contour in contours:
            # print(f"size: {cv2.contourArea(contour)}")
            color = retrieve_area_color(rgb_image, contour, colors_hsv)
            # color = retrieve_center_color(rgb_image, contour, colors_hsv)
            # theta, corners = get_orientation(contour)
            # if theta is None:
            theta = cv2.minAreaRect(contour)[2]
            # cnt = contour
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # cx = cX
            # cy = cY
            # cnt_norm = cnt - [cx, cy]

            # scale = 3000.0
            # cnt_scaled = cnt_norm * scale

            # cnt_scaled = cnt_scaled + [cx, cy]
            # cnt_scaled = cnt_scaled.astype(np.int32)

            # contour = cnt_scaled
            # theta = cv2.minAreaRect(contour)[2]
            
            depth = depth_image[cY, cX]
            cv2.putText(cnt_image, color, (cX-30, cY+40), font, 1.0, (0,0,0), thickness=2)
            cv2.putText(cnt_image, str(int(theta)), (cX, cY), font, 0.5, (255,255,255), thickness=2)
            # Additional parameters ---------------------------------------------------------------
            xy_coords = f"({cX}, {cY})"
            cv2.putText(cnt_image, xy_coords, (cX-30, cY-30), font, 0.5, (20, 255, 57), thickness=2)
            if get_size(contour):
                cv2.putText(cnt_image, "L", (cX-45, cY), font, 1, (255,255,255), thickness=2)
            else:
                cv2.putText(cnt_image, "S", (cX-45, cY), font, 1, (255,255,255), thickness=2)
            if get_shape(contour):
                cv2.putText(cnt_image, "s", (cX+45, cY), font, 1, (255, 255, 0), thickness=2)
            # END Additional parameters
            # -------------------------------------------------------------------------------------
            # print(f"cX = {cX}, cY = {cY}, z = {depth}, color = {color}")
            
            # convert the pixel coordinates to world coordinates
            # intrinsic_vec = np.array([cX, cY, depth])
            
            # append the block detection
            block_detections = np.concatenate([block_detections, np.array([cX, cY, depth, theta])])

        self.block_detections = block_detections.reshape((-1, 4))
        # cv2.imshow("Thresh:", thresh)
        # cv2.imshow("Cleaned:", cleaned)
        # cv2.imshow("Final", cnt_image)

        # cv2.drawContours(cnt_image, contours, -1, (0,255,255), 3)
        # self.VideoFrame = cv2.cvtColor(cnt_image, cv2.COLOR_BGR2RGB)
        self.VideoFrame = cnt_image
        self.block_contours = contours












    def detectBlocksInDepthImage(self):
        """!
        @brief      Detect blocks from depth

                    TODO: Implement a blob detector to find blocks in the depth image
        """
        pass

    def projectGridInRGBImage(self):
        """!
        @brief      projects

                    TODO: Use the intrinsic and extrinsic matricies to project the gridpoints 
                    on the board into pixel coordinates. copy self.VideoFrame to self.GridFrame
                    and draw on self.GridFrame the grid intersection points from self.grid_points
                    (hint: use the cv2.circle function to draw circles on the image)
        """
        if self.camera_calibrated:
            modified_image = self.VideoFrame.copy()
            for i in range(self.grid_points.shape[1]):
                for j in range(self.grid_points.shape[2]):
                    x = int(self.grid_points[0, i, j])
                    y = int(self.grid_points[1, i, j])
                    world_vec = np.array([x, y, 0, 1])
                    # camera_vec = np.dot(self.extrinsic_matrix, world_vec)
                    # # print("camera_vec", camera_vec)
                    # img_vec = np.dot(self.intrinsic_matrix, camera_vec[:3].reshape(3, 1))
                    # # print(img_vec)
                    # img_vec_H = np.dot(self.H, img_vec)
                    # # print(img_vec_H)
                    # img_vec_H = img_vec_H / img_vec_H[2]
                    # x = int(img_vec_H[0])
                    # y = int(img_vec_H[1])
                    pixel = world_to_frame(world_vec, 
                                           self.extrinsic_matrix, 
                                           use_factory=True, 
                                           H=self.H)
                    x = int(pixel[0])
                    y = int(pixel[1])
                    # Draw a green circle of radius 3
                    cv2.circle(modified_image, (x, y), 3, (0, 255, 0), -1)
                    
            self.GridFrame = modified_image
        
    def drawTagsInRGBImage(self, msg):
        """
        @brief      Draw tags from the tag detection

                    TODO: Use the tag detections output, to draw the corners/center/tagID of
                    the apriltags on the copy of the RGB image. And output the video to self.TagImageFrame.
                    Message type can be found here: /opt/ros/humble/share/apriltag_msgs/msg

                    center of the tag: (detection.centre.x, detection.centre.y) they are floats
                    id of the tag: detection.id
        """
        modified_image = self.VideoFrame.copy()
        # Write your code here
        detections = msg.detections
        for detection in detections:
            id = detection.id
            centre = detection.centre
            if self.camera_calibrated:
                fixed_centre = apply_homography(self.H, np.array([centre.x, centre.y]))
                final_centre_x = int(fixed_centre[0])
                final_centre_y = int(fixed_centre[1])
            else:
                final_centre_x = int(centre.x)
                final_centre_y = int(centre.y)
            cv2.circle(modified_image, (final_centre_x, final_centre_y), 5, (0, 255, 0), -1)
            cv2.putText(modified_image, "ID:" + str(id), (final_centre_x + 5, final_centre_y - 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0),
                        2, cv2.LINE_AA)
            corners = detection.corners
            corners_np = np.zeros((4, 2), dtype=np.float64)
            for i, corner in enumerate(corners):
                if self.camera_calibrated:
                    fixed_corner = apply_homography(self.H, np.array([corner.x, corner.y]))
                    corners_np[i] = [fixed_corner[0], fixed_corner[1]]
                else:
                    corners_np[i] = [corner.x, corner.y]
            corners_np = corners_np.reshape((-1, 1, 2))
            corners_np = corners_np.astype(np.int32)
            cv2.polylines(modified_image, [corners_np], True, (0, 0, 255), 2)

        self.TagImageFrame = modified_image


class ImageListener(Node):
    def __init__(self, topic, camera):
        super().__init__('image_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, data.encoding)
        except CvBridgeError as e:
            print(e)
        self.camera.VideoFrame = cv_image
        if self.camera.camera_calibrated is True:
            new_img = cv2.warpPerspective(cv_image, self.camera.H, (1280, 720))
            self.camera.VideoFrame = new_img
        if self.camera.block_detected is True:
            self.camera.transform_images()
            self.camera.blockDetector()
            self.camera.processVideoFrame()

class TagDetectionListener(Node):
    def __init__(self, topic, camera):
        super().__init__('tag_detection_listener')
        self.topic = topic
        self.tag_sub = self.create_subscription(
            AprilTagDetectionArray,
            topic,
            self.callback,
            10
        )
        self.camera = camera

    def callback(self, msg):
        self.camera.tag_detections = msg
        if np.any(self.camera.VideoFrame != 0):
            self.camera.drawTagsInRGBImage(msg)


class CameraInfoListener(Node):
    def __init__(self, topic, camera):
        super().__init__('camera_info_listener')  
        self.topic = topic
        self.tag_sub = self.create_subscription(CameraInfo, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        self.camera.intrinsic_matrix = np.reshape(data.k, (3, 3))
        # print(self.camera.intrinsic_matrix)


class DepthListener(Node):
    def __init__(self, topic, camera):
        super().__init__('depth_listener')
        self.topic = topic
        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image, topic, self.callback, 10)
        self.camera = camera

    def callback(self, data):
        try:
            cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding)
            # cv_depth = cv2.rotate(cv_depth, cv2.ROTATE_180)
        except CvBridgeError as e:
            print(e)
        self.camera.DepthFrameRaw = cv_depth
        # self.camera.DepthFrameRaw = self.camera.DepthFrameRaw / 2
        self.camera.ColorizeDepthFrame()


class VideoThread(QThread):
    updateFrame = pyqtSignal(QImage, QImage, QImage, QImage)

    def __init__(self, camera, parent=None):
        QThread.__init__(self, parent=parent)
        self.camera = camera
        image_topic = "/camera/color/image_raw"
        depth_topic = "/camera/aligned_depth_to_color/image_raw"
        camera_info_topic = "/camera/color/camera_info"
        tag_detection_topic = "/detections"
        image_listener = ImageListener(image_topic, self.camera)
        depth_listener = DepthListener(depth_topic, self.camera)
        camera_info_listener = CameraInfoListener(camera_info_topic,
                                                  self.camera)
        tag_detection_listener = TagDetectionListener(tag_detection_topic,
                                                      self.camera)
        
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(image_listener)
        self.executor.add_node(depth_listener)
        self.executor.add_node(camera_info_listener)
        self.executor.add_node(tag_detection_listener)

    def run(self):
        if __name__ == '__main__':
            cv2.namedWindow("Image window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Tag window", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Grid window", cv2.WINDOW_NORMAL)
            time.sleep(0.5)
        try:
            while rclpy.ok():
                start_time = time.time()
                rgb_frame = self.camera.convertQtVideoFrame()
                depth_frame = self.camera.convertQtDepthFrame()
                tag_frame = self.camera.convertQtTagImageFrame()
                self.camera.projectGridInRGBImage()
                grid_frame = self.camera.convertQtGridFrame()
                if ((rgb_frame != None) & (depth_frame != None)):
                    self.updateFrame.emit(
                        rgb_frame, depth_frame, tag_frame, grid_frame)
                self.executor.spin_once() # comment this out when run this file alone.
                elapsed_time = time.time() - start_time
                sleep_time = max(0.03 - elapsed_time, 0)
                time.sleep(sleep_time)

                if __name__ == '__main__':
                    cv2.imshow(
                        "Image window",
                        cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Depth window", self.camera.DepthFrameRGB)
                    cv2.imshow(
                        "Tag window",
                        cv2.cvtColor(self.camera.TagImageFrame, cv2.COLOR_RGB2BGR))
                    cv2.imshow("Grid window",
                        cv2.cvtColor(self.camera.GridFrame, cv2.COLOR_RGB2BGR))
                    cv2.waitKey(3)
                    time.sleep(0.03)
        except KeyboardInterrupt:
            pass
        
        self.executor.shutdown()
        

def main(args=None):
    rclpy.init(args=args)
    try:
        camera = Camera()
        videoThread = VideoThread(camera)
        videoThread.start()
        try:
            videoThread.executor.spin()
        finally:
            videoThread.executor.shutdown()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()