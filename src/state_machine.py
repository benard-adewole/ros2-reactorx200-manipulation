"""!
The state machine that implements the logic.
"""
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QTimer
import time
import numpy as np
import rclpy
import cv2
from constant import *
from utils import *
from kinematics import IK_geometric, clamp_half_pi
import uuid

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.gripper_state = 1
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,          0.0,        0.0],
            [0.75*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.5*-np.pi/2,   -0.5,      -0.3,      np.pi/2,        0.0],
            [0.25*-np.pi/2,   0.5,       0.3,     -np.pi/3,    np.pi/2],
            [0.0,             0.0,       0.0,          0.0,        0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [0.5*np.pi/2,     0.5,       0.3,     -np.pi/3,        0.0],
            [0.75*np.pi/2,   -0.5,      -0.3,          0.0,    np.pi/2],
            [np.pi/2,         0.5,       0.3,     -np.pi/3,        0.0],
            [0.0,             0.0,       0.0,          0.0,        0.0]]
        self.recorded_waypoints = []


    def set_next_state(self, state):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and functions as needed.
        """

        # IMPORTANT: This function runs in a loop. If you make a new state, it will be run every iteration.
        #            The function (and the state functions within) will continuously be called until the state changes.

        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        # if self.next_state == "teach_and_repeat_idle":
        #     self.teach_and_repeat_idle()
        
        if self.next_state == "teach_and_repeat_initialize_teach":
            self.teach_and_repeat_initialize_teach()

        if self.next_state == "teach_and_repeat_teach":
            self.teach_and_repeat_teach()
        
        if self.next_state == "teach_and_repeat_end_teach":
            self.teach_and_repeat_end_teach()
        
        if self.next_state == "teach_and_repeat_initialize_repeat":
            self.teach_and_repeat_initialize_repeat()
        
        if self.next_state == "teach_and_repeat_repeat":
            self.teach_and_repeat_repeat()
        
        if self.next_state == "save_waypoints":
            self.save_waypoints()

        if self.next_state == "save_open_gripper":
            self.save_open_gripper()

        if self.next_state == "save_closed_gripper":
            self.save_closed_gripper()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()
        
        if self.next_state == "IK_move":
            self.IK_move()
        
        if self.next_state == "click_and_grab_idle":
            self.click_and_grab_idle()

        if self.next_state == "click_and_grab_grab":
            self.click_and_grab_grab()
        
        if self.next_state == "click_and_grab_place_idle":
            self.click_and_grab_place_idle()
        
        if self.next_state == "click_and_grab_place":
            self.click_and_grab_place()
        
        if self.next_state == "competition1":
            self.competition1()
        
        if self.next_state == "competition2":
            self.competition2()
        
        if self.next_state == "competition3":
            self.competition3()

        if self.next_state == "competition3_test":
            self.competition3_test()

        if self.next_state == "snap_picture":
            self.snap_picture()
        
        if self.next_state == "competition4":
            self.competition4()



    """Functions run for each state"""

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.camera.new_click = False
        self.current_state = "idle"

    # def teach_and_repeat_idle(self):
    #     """
    #     @brief      the idle state for the teach and repeat mode
    #     """
    #     self.status_message = "State: Teach and Repeat - Waiting for input"
    #     self.current_state = "teach_and_repeat_idle"

    def teach_and_repeat_initialize_teach(self):
        """
        @brief      initalize the teach mode (only once)
        """
        self.status_message = "State: Teach and Repeat - Initalizing teach mode"
        self.current_state = "teach_and_repeat_initialize_teach"
        # set the gripper to open
        self.rxarm.gripper.release()
        # set the torque off
        self.rxarm.disable_torque()
        # free the memory
        self.recorded_waypoints = []
        self.gripper_state = 1      # open = 1
        
        self.next_state = "teach_and_repeat_teach"
    
    def teach_and_repeat_teach(self):
        """
        @brief      record the waypoints
        """
        self.status_message = "State: Teach and Repeat - Recording waypoints"
        self.current_state = "teach_and_repeat_teach"
        
        # stay in this state until the user chooses to save the waypoint / gripper
        self.next_state = "teach_and_repeat_teach" 

    def teach_and_repeat_end_teach(self):
        """
        @brief      end the teach mode
        """
        self.status_message = "State: Teach and Repeat - Ending teach mode"
        self.current_state = "teach_and_repeat_end_teach"
        # set the torque on
        self.rxarm.enable_torque()
        # self.next_state = "teach_and_repeat_idle"
        # Optionally, we can append a "final state" to the waypoints to make it back to the default pose
        self.next_state = "idle"
    
    def save_waypoints(self):
        new_pos = self.rxarm.get_positions()
        merged_pos = np.zeros(6)
        merged_pos[:5] = new_pos
        merged_pos[-1] = self.gripper_state
        self.recorded_waypoints.append(merged_pos)
        print("Current recorded waypoints: ", self.recorded_waypoints)
        self.status_message = "State: Teach and Repeat - Saving Waypoints"
        self.current_state = "save_waypoints"
        self.next_state = "teach_and_repeat_teach"
    
    def save_open_gripper(self):
        self.status_message = "State: Teach and Repeat - Saving Open gripper"
        self.current_state = "save_open_gripper"
        new_pos = self.rxarm.get_positions()
        self.gripper_state = 1
        merged_pos = np.zeros(6)
        merged_pos[:5] = new_pos
        merged_pos[-1] = self.gripper_state
        self.recorded_waypoints.append(merged_pos)
        print("Current recorded waypoints: ", self.recorded_waypoints)
        self.next_state = "teach_and_repeat_teach"
        

    def save_closed_gripper(self):
        self.status_message = "State: Teach and Repeat - Saving Closed gripper"
        self.current_state = "save_closed_gripper"
        new_pos = self.rxarm.get_positions()
        self.gripper_state = 0
        merged_pos = np.zeros(6)
        merged_pos[:5] = new_pos
        merged_pos[-1] = self.gripper_state
        self.recorded_waypoints.append(merged_pos)
        print("Current recorded waypoints: ", self.recorded_waypoints)
        self.next_state = "teach_and_repeat_teach"
        
    
    def teach_and_repeat_initialize_repeat(self):
        """
        @brief      initalize the repeat mode
        """
        self.status_message = "State: Teach and Repeat - Initalizing repeat mode"
        self.current_state = "teach_and_repeat_initialize_repeat"
        
        # initialize the arm
        self.rxarm.initialize()
        self.rxarm.gripper.release()
        time.sleep(self.rxarm.get_total_move_time())
        self.next_state = "teach_and_repeat_repeat"
    
    def teach_and_repeat_repeat(self):
        """
        @brief      repeat the waypoints
        """
        self.status_message = "State: Teach and Repeat - Repeating waypoints"
        self.current_state = "teach_and_repeat_repeat"
        
        gripper_state = 1 # 1 means open (maybe, need to check)
        # iterate through the waypoints
        for waypoint in self.recorded_waypoints:
            # suppose the waypoint here is a list of 6 elements, and the last element is the gripper state
            print("waypoints =", waypoint)
            cur_gripper_state = waypoint[5]
            self.rxarm.set_positions(waypoint[:5])
            if cur_gripper_state != gripper_state:
                time.sleep(self.rxarm.get_total_move_time())
                self.rxarm.gripper.release() if cur_gripper_state == 1 else self.rxarm.gripper.grasp()
                gripper_state = cur_gripper_state
            time.sleep(self.rxarm.get_total_move_time())

        self.next_state = "idle"


    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.current_state = "execute"
        for waypoint in self.waypoints:
            if self.current_state == "estop":
                break
            self.rxarm.set_positions(waypoint)
            time.sleep(self.rxarm.get_total_move_time())
        
        self.rxarm.sleep()

        self.status_message = "State: Execute - Executing motion plan"
        self.next_state = "idle"

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        self.status_message = "State: Calibrate - Calibrating"
        distCoeffs = FACTORY_RGB_DISTCOEFFS
        cameraMatrix = FACTORY_RGB_INTRINSIC_MATRIX
        detections = self.camera.tag_detections.detections

        camera_points = np.zeros((len(detections), 2), dtype=np.float32)
        for i, detection in enumerate(detections):
            camera_points[i][0] = detection.centre.x
            camera_points[i][1] = detection.centre.y

        world_points = np.array(self.camera.tag_locations)
        # append the z value (0) to world_points
        world_points = np.append(world_points, np.zeros((4, 1)), axis=1)
        added_tags = np.array([[-425, -100, 155], [425, -100, 92], [425, 400, 155], [-425, 400, 155]], dtype=np.float32)
        world_points = np.vstack((world_points, added_tags))
        
        successful, rvec, tvec = cv2.solvePnP(world_points, camera_points, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(rvec)
        print("Was Successful? ", successful)
        extrinsic_matrix = np.row_stack((np.column_stack((R, tvec)), (0, 0, 0, 1)))

        self.camera.intrinsic_matrix = cameraMatrix
        self.camera.extrinsic_matrix = extrinsic_matrix
        self.camera.camera_calibrated = True

        detections = self.camera.tag_detections.detections
        camera_points = np.zeros((4, 2), dtype=np.float32)
        for i, detection in enumerate(detections[:4]):
            camera_points[i][0] = detection.centre.x
            camera_points[i][1] = detection.centre.y
        dest_points = [
            [380, 525],         # Tag 1 Modified
            [900, 525],         # Tag 2 Modified
            [900, 213],         # Tag 3 Modified
            [380, 213],         # Tag 4 Modified
        ]
        H = cv2.findHomography(camera_points, np.array(dest_points))[0]
        # print(H)
        self.camera.H = H
        print("camera calibrated", self.camera.camera_calibrated)

        self.status_message = "Calibration - Completed Calibration"

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """
        self.current_state = "detect"
        self.status_message = "State: Detect - Detecting blocks"
        self.camera.block_detected = True

        self.next_state = "idle"

    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            time.sleep(5)
        self.next_state = "idle"
    
    def IK_move(self):
        """!
        @brief      Move the arm to a specific position using inverse kinematics
        """
        self.current_state = "IK_move"
        self.status_message = "State: Inverse Kinematics - Moving to position"
        x, y, z, psi = 0, 0, 0, 0
        # x, y, z, psi = 0, 424.15, 303.91, 0
        x, y, z, psi = 100, 225, 120, 90 / 180 * np.pi

        xc  = x - 174.15 * np.cos(psi) * x / np.sqrt(x ** 2 + y ** 2)
        yc  = y - 174.15 * np.cos(psi) * y / np.sqrt(x ** 2 + y ** 2)
        zc  = z + 174.15 * np.sin(psi) 

        print(f"xc: {xc}, yc: {yc}, zc: {zc}")

        y_p = np.sqrt(xc ** 2 + yc ** 2)
        z_p = zc - 103.91

        q1 = np.arctan2(-xc, yc)
        q2 = np.arctan2(y_p, z_p) - np.arccos((2500 + y_p ** 2 + (z_p) ** 2) / (100 * np.sqrt(17 * (y_p ** 2 + z_p ** 2)))) - np.arctan2(1, 4)
        q3 = np.arccos((y_p ** 2 + z_p ** 2 - 82500) / (20000 * np.sqrt(17))) - np.arctan2(4, 1)

        q4 = psi - q2 - q3
        print(f"q1 {q1}, q2 {q2}, q3 {q3}, q4 {q4}")
        self.rxarm.set_positions([q1, q2, q3, q4, 0])
        time.sleep(self.rxarm.get_total_move_time())
        self.next_state = "idle"

    def click_and_grab_idle(self):
        """!
        @brief      Click and grab the block
        """
        self.current_state = "click_and_grab_idle"
        self.status_message = "State: Click and Grab Idle - Waiting for the first click to grab the object"
        # print("new click", self.camera.new_click)
        if self.camera.new_click == True:
            self.next_state = "click_and_grab_grab"
        else:
            self.next_state = "click_and_grab_idle"

    def click_and_grab_grab(self):
        """!
        @brief      Grab the block
        """
        self.current_state = "click_and_grab_grab"
        self.status_message = "State: Click and Grab Grab - Grabbing the object"
        blocks = self.camera.block_detections.copy()
        pt = self.camera.last_click
        closest_block = get_closest_block(blocks, np.array([pt[0], pt[1]]))
        assert closest_block is not None
        x, y, z, orientation = closest_block
        print("Block detected at: ", x, y, z, orientation)

        # Get the world coordinates of the block
        frame_vec = np.array([x, y, 1], dtype=np.float32)
        world_vec = frame_to_world(frame_vec,
                                    self.camera.DepthFrameRaw,
                                    self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H,
                                    use_post_offset_adjustment=True)
        print("World coordinates: ", world_vec)
        
        Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]

        # Get the trajectory to grab the block and also check the doability in advance before the actual execution
        trajs = get_traj_grab([Xw, Yw, Zw], orientation)
        if trajs is None:
            print("Cannot grab the block, please choose another block or adjust the block's position and try again")
            self.next_state = "click_and_grab_idle"
            self.camera.new_click = False
            return
        for traj in trajs:
            if isinstance(traj[0], str):
                self.rxarm.gripper.grasp()
            else:
                self.rxarm.set_positions(traj[0])
            time.sleep(self.rxarm.get_total_move_time())
        self.camera.new_click = False
        self.next_state = "click_and_grab_place_idle"
    
    def click_and_grab_place_idle(self):
        """!
        @brief      Waiting for the second click to place the object
        """
        self.current_state = "click_and_grab_place_idle"
        self.status_message = "State: Click and Grab Place Idle - Waiting for the second click to place the object"
        if self.camera.new_click == True:
            self.next_state = "click_and_grab_place"
        else:
            self.next_state = "click_and_grab_place_idle"


    def click_and_grab_place(self):
        """!
        @brief      Place the object
        """
        self.current_state = "click_and_grab_place"
        self.status_message = "State: Click and Grab - Placing the object"
        x, y = self.camera.last_click

        # Get the world coordinates of the block
        frame_vec = np.array([x, y, 1], dtype=np.float32)
        world_vec = frame_to_world(frame_vec,
                                    self.camera.DepthFrameRaw,
                                    self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H,
                                    use_post_offset_adjustment=True)
        print("World coordinates: ", world_vec)
        Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
        
        # Get the trajectory to place the block
        trajs = get_traj_place([Xw, Yw, Zw], orthogonal_orientation=True)
        if trajs is None:
            print("Cannot place the block, please choose another position and try again")
            self.next_state = "click_and_grab_place_idle"
            self.camera.new_click = False
            return
        for traj in trajs:
            if isinstance(traj[0], str):
                self.rxarm.gripper.release()
            else:
                self.rxarm.set_positions(traj[0])
            time.sleep(self.rxarm.get_total_move_time())
        
        self.camera.new_click = False

        self.next_state = "idle"
    
    def competition1(self):
        """!
        @brief      Competition 1
        """
        self.current_state = "competition1"
        self.status_message = "State: Competition 1 - Running Competition 1"
        
        # First unstacked the blocks, clean the table
        self.rxarm.set_straight_up()
        time.sleep(1.5)

        while True:
            print("unstacking")
            if self.current_state == "idle":
                break
            # find the block that has the smallest depth (highest)
            highest_block = self.camera.block_detections[0]
            count_cube = 0
            for i, detection in enumerate(self.camera.block_detections):
                print(f"detected {i}:", detection)
                contour = self.camera.block_contours[i]
                if get_shape(contour):
                    count_cube += 1
                # if not get_shape(contour):
                #     # filter out non-cube blocks
                #     print("get filtered out", detection)
                #     continue
                if highest_block[2] > detection[2]:
                    highest_block = detection
            self.status_message = f"State: Competition 1 - Clearing stacked block"
            if count_cube == 6:
                break
            # grab it somewhere
            x, y, _, orientation = highest_block
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw + 2], orientation)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            # Place the block
            while len(LOADING_POINTS):
                world_x, world_y = LOADING_POINTS.pop(0)
                if check_clean(world_x, world_y, self.camera.DepthFrameWarped, self.camera.extrinsic_matrix, self.camera.H):
                    break
            x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                  extrinsic_matrix=self.camera.extrinsic_matrix,
                                  use_factory=True,
                                  H=self.camera.H)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to place the block
            trajs = get_traj_place([Xw, Yw, Zw], straight_orientation=True)
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.release()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
        
        # Clear the two target positions
        for point in [SMALL_BLOCK_POS, LARGE_BLOCK_POS]:
            if check_clean(point[0], point[1], self.camera.DepthFrameWarped, self.camera.extrinsic_matrix, self.camera.H):
                print(f"point {point} is clean")
                continue
            frame_x, frame_y, _ = world_to_frame(world_vec=np.array([point[0], point[1], 0, 1]),
                                  extrinsic_matrix=self.camera.extrinsic_matrix,
                                  use_factory=True,
                                  H=self.camera.H)
            # Get the closest contour to the target position
            closest_block = get_closest_block(self.camera.block_detections, np.array([frame_x, frame_y]))
            x, y, _, orientation = closest_block
            print("block that needs to be cleared from the target positions", closest_block)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw], orientation)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            # Place the block
            while len(LOADING_POINTS):
                world_x, world_y = LOADING_POINTS.pop(0)
                if check_clean(world_x, world_y, self.camera.DepthFrameWarped, self.camera.extrinsic_matrix, self.camera.H):
                    break
            x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                  extrinsic_matrix=self.camera.extrinsic_matrix,
                                  use_factory=True,
                                  H=self.camera.H)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to place the block
            # trajs = get_traj_place([Xw, Yw, Zw], straight_orientation=True)
            trajs = get_traj_place([Xw, Yw, Zw])
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.release()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
        
        # Detect the blocks with multiple votes
        all_detections = []
        for i in range(MAJORITY_VOTES_K):
            block_contours = self.camera.block_contours.copy()
            block_detections = self.camera.block_detections.copy()
            videoframe = self.camera.VideoFrame.copy()
            sorted_small_blocks, sorted_large_blocks = sort_blocks_with_size(block_contours, videoframe, block_detections)
            # sorted_small_blocks and sorted_large_blocks each look like:
            # [([x, y, depth, orientation], color), ...]
            all_detections.append((sorted_small_blocks, sorted_large_blocks))
            time.sleep(0.1)
        small_blocks, large_blocks = majority_vote_aggregate(all_detections)

        print("#" * 80)
        print("final small blocks:", small_blocks)
        print("final large blocks:", large_blocks)
        
        # Start from sorting the small blocks
        while len(small_blocks):
            if self.current_state == "idle":
                break
            block, color = small_blocks.pop(0)
            # Get the world coordinates of the block
            self.status_message = f"State: Competition 1 - Grabbing small block {color}"
            x, y, _, orientation = block
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw], orientation)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            # Place the block
            print("Placing small blocks")
            self.status_message = f"State: Competition 1 - Placing small block {color}"
            world_x, world_y = SMALL_BLOCK_POS
            x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                  extrinsic_matrix=self.camera.extrinsic_matrix,
                                  use_factory=True,
                                  H=self.camera.H)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print(f"Placing small {color} block to:", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            Yw -= (2 - len(small_blocks)) * 5
            Xw -= (2 - len(small_blocks)) * 4
            if len(small_blocks) == 0:
                Zw += 15
            # Get the trajectory to place the block
            # trajs = get_traj_place([Xw, Yw, Zw], straight_orientation=True, is_small=True)
            trajs = get_traj_place([Xw, Yw, Zw], is_small=True)
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.release()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())


        # Start from sorting the large blocks
        print("Placing large blocks")
        while len(large_blocks):
            if self.current_state == "idle":
                break
            block, color = large_blocks.pop(0)
            # Get the world coordinates of the block
            self.status_message = f"State: Competition 1 - Grabbing large block {color}"
            x, y, _, orientation = block
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw], orientation)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            # Place the block
            self.status_message = f"State: Competition 1 - Placing large block {color}"
            world_x, world_y = LARGE_BLOCK_POS
            x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                  extrinsic_matrix=self.camera.extrinsic_matrix,
                                  use_factory=True,
                                  H=self.camera.H)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print(f"Placing large {color} block to:", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            Xw += (2 - len(large_blocks)) * 5
            Yw -= (2 - len(large_blocks)) * 8
            if len(large_blocks) == 0:
                Zw += 25
            # Get the trajectory to place the block
            trajs = get_traj_place([Xw, Yw, Zw], is_small=False)
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.release()
                else:
                    print(traj[0])
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())

        self.rxarm.initialize()
        time.sleep(self.rxarm.get_total_move_time())
        
        self.next_state = "idle"
    
    def competition2(self):
        """!
        @brief      Competition 2
        """
        self.current_state = "competition2"
        self.status_message = "State: Competition 2 - Running Competition 2"
        self.rxarm.set_straight_up()
        time.sleep(1.5)
        # Unstack the blocks until there are 12 cubes

        while True:
            print("############## UNSTACKING ##############")
            
            count_cube = 0
            block_detections = self.camera.block_detections.copy()
            videoframe = self.camera.VideoFrame.copy()
            count_trial = 0

            while count_trial < 10:
                count_cube = 0
                for i, j in enumerate(block_detections):
                    print(f"detected {i}:", j)
                    contour = self.camera.block_contours[i]
                    if get_shape(contour):
                        count_cube += 1
                if count_cube == 12:
                    break
                count_trial += 1
                time.sleep(0.1)
                block_detections = self.camera.block_detections.copy()
                videoframe = self.camera.VideoFrame.copy()
                print("count_trial", count_trial, "count_cube", count_cube)
            if count_cube == 12:
                break
            block_contours = self.camera.block_contours.copy()
            depth_warped = self.camera.DepthFrameWarped.copy()
            
            highest_block = self.camera.block_detections[0]
            highest_contour = self.camera.block_contours[0]
            min_depth = np.inf
            for i, detection in enumerate(block_detections):
                contour = block_contours[i]
                depth = get_90per_depth(depth_warped, contour)
                # if get_shape(contour):
                #     count_cube += 1
                #     print(f"detected {i}:", detection, count_cube, "size=", get_size(contour), "color", retrieve_area_color(videoframe, contour, colors_hsv))
                y = detection[1]
                if min_depth > depth:   # avoid the blocks on the bottom half of the screen; this avoids keep detecting a tall cylinder as the highest block
                    highest_block = detection
                    highest_contour = contour
                    min_depth = depth
            # if count_cube == 12:
            #     break
            self.status_message = f"State: Competition 2 - Clearing stacked block"
            # grab it somewhere
            cur_size = get_size(highest_contour)
            x, y, _, orientation = highest_block
            # print("chosen block", highest_block)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            # print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw], orientation)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            # Place the block
            while len(LOADING_POINTS_COMPETITION_2):
                world_x, world_y = LOADING_POINTS_COMPETITION_2.pop(0)
                if check_clean(world_x, world_y, self.camera.DepthFrameWarped, self.camera.extrinsic_matrix, self.camera.H):
                    break
            x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                    self.camera.DepthFrameRaw,
                                    self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H,
                                    use_post_offset_adjustment=True)
            print(f"Placing block to:", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to place the block
            trajs = get_traj_place([Xw, Yw, Zw], straight_orientation=True, is_small=not cur_size)
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.release()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
        area_min_x, area_max_y = LOADING_AREA_CORNER['top_left']
        area_max_x, area_min_y = LOADING_AREA_CORNER['bottom_right']
        frame_min_x, frame_min_y, _ = world_to_frame(world_vec=np.array([area_min_x, area_max_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
        frame_max_x, frame_max_y, _ = world_to_frame(world_vec=np.array([area_max_x, area_min_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
        print(f"frame range: x in ({frame_min_x}, {frame_max_x}), y in ({frame_min_y}, {frame_max_y})")
        while True:
            all_clear = True
            x, y, orientation = 0, 0, 0
            for block in self.camera.block_detections:
                x, y, _, orientation = block
                if x > frame_min_x and x < frame_max_x and y > frame_min_y and y < frame_max_y:
                    all_clear = False
                    break
            if all_clear:
                break
            # Move the block to the loading area
            print("########### Clearing Area ##############")
            # print(f"clearing {x}, {y}")
            self.status_message = f"State: Competition 2 - Clearing area"
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            # print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw], orientation)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            
            # Place the block
            while len(LOADING_POINTS_COMPETITION_2):
                world_x, world_y = LOADING_POINTS_COMPETITION_2.pop(0)
                depth_warped = self.camera.DepthFrameWarped.copy()
                if check_clean(world_x, world_y, depth_warped, self.camera.extrinsic_matrix, self.camera.H):
                    break
            x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                    self.camera.DepthFrameRaw,
                                    self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H,
                                    use_post_offset_adjustment=True)
            # print(f"Placing block to:", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to place the block
            trajs = get_traj_place([Xw, Yw, Zw], straight_orientation=True, )
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.release()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())

        # Now start the competition
        waiting_blocks = []
        for size in ['small', 'large']:
            for color in ['red', 'orange', 'yellow', 'green', 'blue', 'violet']:
                waiting_blocks.append(f"{size}_{color}")
        full_blocks = waiting_blocks.copy()
        print("#### start competition ####")
        while len(waiting_blocks):
            current_block_desc, current_block, current_contour = None, None, None
            block_contours, block_detections, videoframe = None, None, None
            while True:
                while True:
                    # detect the color until we see exact 12 blocks with expected colors and sizes
                    block_contours = self.camera.block_contours.copy()
                    block_detections = self.camera.block_detections.copy()
                    videoframe = self.camera.VideoFrame.copy()
                    
                    copy_waiting_blocks = full_blocks.copy()
                    for contour in block_contours:
                        if not get_shape(contour):
                            # filter out non-cube blocks
                            continue
                        size = 'large' if get_size(contour) else 'small'
                        color = retrieve_area_color(videoframe, contour, colors_hsv)
                        block_desc = f"{size}_{color}"
                        # print(block_desc)
                        if block_desc in copy_waiting_blocks:
                            copy_waiting_blocks.remove(block_desc)
                    if len(copy_waiting_blocks) == 0:
                        break
                    print("Not detected blocks:", copy_waiting_blocks)
                    time.sleep(0.1)

                current_block_desc = waiting_blocks[0]
                for contour, detection in zip(block_contours, block_detections):
                    if not get_shape(contour):
                        # filter out non-cube blocks
                        continue
                    size = 'large' if get_size(contour) else 'small'
                    color = retrieve_area_color(self.camera.VideoFrame, contour, colors_hsv)
                    block_desc = f"{size}_{color}"
                    if block_desc != current_block_desc:
                        continue
                    current_contour = contour
                    current_block = detection
                
                x, y, _, orientation = current_block
                if x > frame_min_x and x < frame_max_x and y > frame_min_y and y < frame_max_y:
                    # grabbing a block from the working area, which means a wrong move
                    continue
                else:
                    break
                

            print(f"State: Competition 2 - Grabbing {current_block_desc}")
            x, y, _, orientation = current_block
            cur_size = get_size(current_contour)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw], orientation)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            # Place the block
            self.status_message = f"State: Competition 2 - Placing {current_block_desc}"
            world_x, world_y = BLOCK_TO_POS[current_block_desc]
            x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                  extrinsic_matrix=self.camera.extrinsic_matrix,
                                  use_factory=True,
                                  H=self.camera.H)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print(f"Placing {current_block_desc} block to:", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to place the block
            trajs = get_traj_place([Xw, Yw, Zw], orthogonal_orientation=True, is_small=not cur_size)
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.release()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            
            waiting_blocks.pop(0)

        self.rxarm.initialize()
        time.sleep(self.rxarm.get_total_move_time())
        
        self.next_state = "idle"

    def competition3(self):
        self.current_state = "competition3"
        self.status_message = "State: Competition 3 - Running Competition 3"

        self.rxarm.set_straight_up()
        time.sleep(3)
        area_min_x, area_max_y = LOADING_AREA_CORNER_COMPETITION_3['top_left']
        area_max_x, area_min_y = LOADING_AREA_CORNER_COMPETITION_3['bottom_right']
        frame_min_x, frame_min_y, _ = world_to_frame(world_vec=np.array([area_min_x, area_max_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
        frame_max_x, frame_max_y, _ = world_to_frame(world_vec=np.array([area_max_x, area_min_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
        
        count = 0
        print(f"frame range: x in ({frame_min_x}, {frame_max_x}), y in ({frame_min_y}, {frame_max_y})")
        while True:
            all_clear = True
            x, y, orientation = 0, 0, 0
            for block in self.camera.block_detections:
                x, y, _, orientation = block
                if (x > frame_min_x and x < frame_max_x and y > frame_min_y and y < frame_max_y):
                    print("block detected")
                    break
            if all_clear:
                # no blocks in the loading area, which means the end of the game
                print("block not detected")
                break
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw], orientation)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            # Place the block
            world_x, world_y = STACK_POS
            x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                  extrinsic_matrix=self.camera.extrinsic_matrix,
                                  use_factory=True,
                                  H=self.camera.H)
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            if USE_REF_TABLE:
                Zw = (Zw + REF_TABLE[count]) / 2
            # Get the trajectory to place the block
            trajs = get_traj_place([Xw, Yw, Zw], straight_orientation=True, is_small=False)
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.release()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            count += 1

        self.rxarm.initialize()
        time.sleep(self.rxarm.get_total_move_time())
        self.next_state = "idle"

    def competition3_test(self):
        self.current_state = "competition3_test"
        self.status_message = "State: Competition 3 test - Running Competition 3"

        self.rxarm.set_straight_up()
        time.sleep(3)
        area_min_x, area_max_y = LOADING_AREA_CORNER_COMPETITION_3['top_left']
        area_max_x, area_min_y = LOADING_AREA_CORNER_COMPETITION_3['bottom_right']
        frame_min_x, frame_min_y, _ = world_to_frame(world_vec=np.array([area_min_x, area_max_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
        frame_max_x, frame_max_y, _ = world_to_frame(world_vec=np.array([area_max_x, area_min_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
        
        count = 0
        # trajs_blocks = [(2, 221, -6),
        #              (2, 220, 33),
        #              (2, 215, 72),
        #              (2, 211, 110),
        #              (2, 207, 149.5)]
        print(f"frame range: x in ({frame_min_x}, {frame_max_x}), y in ({frame_min_y}, {frame_max_y})")
        # while len(trajs_blocks):
        while True:
            all_clear = True
            x, y, orientation = 0, 0, 0
            for block in self.camera.block_detections:
                x, y, _, orientation = block
                if (x > frame_min_x and x < frame_max_x and y > frame_min_y and y < frame_max_y):
                    print("block detected")
                    all_clear = False
                    break
            if all_clear:
                # no blocks in the loading area, which means the end of the game
                print("block not detected")
                break
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                       self.camera.DepthFrameRaw,
                                       self.camera.extrinsic_matrix,
                                       use_factory=True,
                                       H=self.camera.H,
                                       use_post_offset_adjustment=True)
            print("World coordinates: ", world_vec)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # Get the trajectory to grab the block and also check the doability in advance before the actual execution
            trajs = get_traj_grab([Xw, Yw, Zw], orientation)
            print(trajs)
            if trajs is None:
                print("Cannot grab the block, please choose another block or adjust the block's position and try again")
                continue
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            # Place the block
            # world_x, world_y = STACK_POS
            # x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
            #                       extrinsic_matrix=self.camera.extrinsic_matrix,
            #                       use_factory=True,
            #                       H=self.camera.H)
            # frame_vec = np.array([x, y, 1], dtype=np.float32)
            # world_vec = frame_to_world(frame_vec,
            #                            self.camera.DepthFrameRaw,
            #                            self.camera.extrinsic_matrix,
            #                            use_factory=True,
            #                            H=self.camera.H,
            #                            use_post_offset_adjustment=True)
            # Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            # print(f"placing it to world: {Xw}, {Yw}, {Zw}")
            # if USE_REF_TABLE:
            #     Zw = (Zw + REF_TABLE[count]) / 2
            # Xw = 2
            # Yw = 221
            # Zw = -5
            # Xw = 2
            # Yw = 220
            # Zw = 33
            # Xw = 2
            # Yw = 215
            # Zw = 72
            # Xw = 2
            # Yw = 211
            # Zw = 110
            # Xw = 2
            # Yw = 207
            # Zw = 149.5
            
            Xw = 383
            Yw = -12
            Zw = 2
            # Xw, Yw, Zw = trajs_blocks.pop(0)
            # place_obj = [-1.60761189,  0.42184472,  0.71636903, -0.79153413,  0.02914564,]
            # self.rxarm.set_positions(place_obj)
            # time.sleep(self.rxarm.get_total_move_time())
            # Get the trajectory to place the block
            trajs = get_traj_place([Xw, Yw, Zw], is_small=False)
            print(trajs)
            for traj in trajs:
                if isinstance(traj[0], str):
                    break
                    self.rxarm.gripper.release()
                else:
                    self.rxarm.set_positions(traj[0])
                time.sleep(self.rxarm.get_total_move_time())
            
            count += 1

        self.next_state = "idle"
    
    def competition4(self):
        """
        For the draft version, we have 3 steps in general:
        1. use the pole to reload the launcher (need a complete list of waypoints)
        2. detect, pick up, and drop the ball (need at least one waypoint that drops the ball)
        3. press the lever to shoot the ball
        """
        self.current_state = "competition4"
        self.status_message = "State: Competition 4 - Running Competition 4"
        
        # Set up the mask for storage. The mask is used to filter out the area that is not related to the storage area for picking up the ball. No need to modify this part unless some detection issue happens.
        area_min_x, area_max_y = STORAGE_CORNER['top_left']
        area_max_x, area_min_y = STORAGE_CORNER['bottom_right']
        frame_min_x, frame_min_y, _ = world_to_frame(world_vec=np.array([area_min_x, area_max_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
        frame_max_x, frame_max_y, _ = world_to_frame(world_vec=np.array([area_max_x, area_min_y, 0, 1]),
                                    extrinsic_matrix=self.camera.extrinsic_matrix,
                                    use_factory=True,
                                    H=self.camera.H)
        

        # 1. Reload the launcher
        # Make sure you get the whole process, including picking up the pole, stab it into the launcher, and place the pole back.
        self.status_message = "State: Competition 4 - Reloading the launcher"
        # for i, waypoint in enumerate(RELOAD_LAUNCHER): # Define the RELOAD_LAUNCHER in src/constant.py
        #     print("waypoint=", waypoint)
        #     if isinstance(waypoint, str):
        #         if waypoint == "release":
        #             self.rxarm.gripper.release()
        #         else:
        #             self.rxarm.gripper.grasp()
        #     else:
        #         self.rxarm.set_positions(waypoint)
        #     time.sleep(self.rxarm.get_total_move_time())
        #     # if i > 6:
        #     #     break
        while True:
            self.rxarm.set_straight_up()
            time.sleep(1)
            # # 2. Detect, pick up, and drop the ball
            x, y, orientation = 0, 0, 0

            while True:
                block_detections = self.camera.block_detections
                has_obj = False
                for block in block_detections:
                    x, y, _, orientation = block
                    if (x > frame_min_x and x < frame_max_x and y > frame_min_y and y < frame_max_y):
                        print("ball detected")
                        has_obj = True
                        break
                if has_obj:
                    break
                time.sleep(1)
            
            # pick up the ball
            frame_vec = np.array([x, y, 1], dtype=np.float32)
            world_vec = frame_to_world(frame_vec,
                                        self.camera.DepthFrameRaw,
                                        self.camera.extrinsic_matrix,
                                        use_factory=True,
                                        H=self.camera.H,
                                        use_post_offset_adjustment=True)
            Xw, Yw, Zw = world_vec[0], world_vec[1], world_vec[2]
            trajs = get_traj_grab([Xw, Yw, Zw - 3], orientation)
            for traj in trajs:
                if isinstance(traj[0], str):
                    self.rxarm.gripper.grasp()
                else:
                    self.rxarm.set_positions(traj[0], is_comp4=True)
                time.sleep(self.rxarm.get_total_move_time() - 0.2)
            
            # # drop the ball
            self.rxarm.set_positions(DROP_BALL, is_comp4=True)
            time.sleep(self.rxarm.get_total_move_time() - 0.3)
            self.rxarm.gripper.release()
            time.sleep(0.5)
            self.rxarm.set_straight_up()
            time.sleep(0.5)
            # self.rxarm.initialize()
            # time.sleep(self.rxarm.get_total_move_time())
            
            # # 3. Press the lever to shoot the ball
            self.status_message = "State: Competition 4 - Pressing the lever"
            self.rxarm.gripper.grasp()
            time.sleep(0.5)
            for waypoint in PRESS_LEVER: # Define the PRESS_LEVER in src/constant.py
                if isinstance(waypoint, tuple):
                    # move slow for push
                    self.rxarm.set_positions(waypoint[0])
                else:
                    self.rxarm.set_positions(waypoint, is_comp4=True)
                time.sleep(self.rxarm.get_total_move_time() - 0.2)
            
            self.rxarm.gripper.release()
            time.sleep(0.5)
            # self.rxarm.initialize()
            # time.sleep(1)

        self.next_state = "idle"
        


    def snap_picture(self):
        self.current_state = "snap_picture"
        self.status_message = "Status: Snap Picture, taking a picture of the board"
        filename = f"output_{uuid.uuid4().hex}.png"
        cv2.imwrite(filename, cv2.cvtColor(self.camera.VideoFrame, cv2.COLOR_BGR2RGB), )
        self.next_state = "idle"


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)
    
    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            time.sleep(0.05)