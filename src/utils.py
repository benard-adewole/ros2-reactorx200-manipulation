import cv2
import numpy as np

from constant import *
from kinematics import IK_geometric, clamp_half_pi, clamp

def apply_homography(H, pixel):
    """
    Add a homography to an original pixel in homogeneous coordinates.
    pixel: np.array with shape (2,) or (3,)
    
    return: normalized pixel in homogeneous coordinates with shape (3,)
    """
    if pixel.shape[0] == 2:
        pixel = np.hstack([pixel, 1])
    vec = H @ pixel
    return vec / vec[2]

def undo_homography(H, pixel):
    """
    Undo a homography from a transformed pixel in homogeneous coordinates.
    pixel: np.array with shape (2,) or (3,)

    return: original pixel in homogeneous coordinates with shape (3,)
    """
    if pixel.shape[0] == 2:
        pixel = np.hstack([pixel, 1])
    vec = np.linalg.inv(H) @ pixel
    return vec / vec[2]

def post_offset_adjustment(world_vec):
    world_vec[2] -= OFFSET_A * world_vec[1] + OFFSET_B
    return world_vec

def frame_to_world(pixel, 
                   depth_data,
                   extrinsic_matrix: np.array,
                   use_factory : bool = True, 
                   H: np.array = None,
                   use_post_offset_adjustment: bool = True):
    """
    Convert a pixel in the frame to a pixel in the world.
    pixel: np.array with shape (2,) or (3,)
    z: depth of the pixel
    extrinsic_matrix: extrinsic matrix of the camera
    use_factory: whether to use the factory calibration
    H: homography matrix if the pixel has been transformed, otherwise None

    return: pixel in the world with shape (3,)
    """
    if use_factory:
        intrinsic_matrix = FACTORY_RGB_INTRINSIC_MATRIX
    else:
        intrinsic_matrix = AVG_INTRINSIC_MATRIX
    
    if H is not None:
        # undo homography
        pixel = undo_homography(H, pixel)
    
    z = depth_data[int(pixel[1]), int(pixel[0])]
    camera_vec = np.matmul(np.linalg.inv(intrinsic_matrix), pixel) * z
    camera_vec = np.append(camera_vec, 1)
    camera_vec = np.transpose(camera_vec)

    world_vec = np.matmul(np.linalg.inv(extrinsic_matrix), camera_vec)

    if use_post_offset_adjustment:
        world_vec = post_offset_adjustment(world_vec)
    
    return world_vec

def world_to_frame(world_vec,
                   extrinsic_matrix: np.array,
                   use_factory: bool = True,
                   H: np.array = None,
):
    """
    Convert a pixel in the world to a pixel in the frame.
    world_vec: np.array with shape (3,) or (4,) in homogeneous coordinates
    extrinsic_matrix: extrinsic matrix of the camera
    use_factory: whether to use the factory calibration
    H: homography matrix if the pixel has been transformed, otherwise None

    return: pixel in the frame with shape (3,)
    """
    if world_vec.shape[0] == 3:
        world_vec = np.append(world_vec, 1)
    camera_vec = np.matmul(extrinsic_matrix, world_vec)
    camera_vec = camera_vec / camera_vec[2]
    
    if use_factory:
        intrinsic_matrix = FACTORY_RGB_INTRINSIC_MATRIX
    else:
        intrinsic_matrix = AVG_INTRINSIC_MATRIX

    pixel = np.matmul(intrinsic_matrix, camera_vec[:3])
    pixel = pixel[:2]
    
    if H is not None:
        # apply homography
        pixel = apply_homography(H, pixel)
    
    return pixel



def retrieve_area_color(data, contour, labels):
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    shifted_contour = contour - np.array([cX, cY])
    shifted_contour = (shifted_contour * 0.1).astype(np.int32)
    shrinked_contour = shifted_contour + np.array([cX, cY])

    mask = np.zeros(data.shape[:2], dtype="uint8")
    
    cv2.drawContours(mask, [shrinked_contour], -1, 255, -1)
    hsv_data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
    # print(contour.shape)
    # print(contour.shape)
    # print(cv2.mean(data, mask=mask))
    mean = cv2.mean(hsv_data, mask=mask)[:3]
    # print(mean)
    min_dist = (np.inf, None)
    for label in labels:
        # d = np.linalg.norm(label["color"] - np.array(mean))
        d = abs(label['color'] - mean[0])
        # print(label["id"], d)
        if d < min_dist[0]:
            min_dist = (d, label["id"])
    return min_dist[1]


def retrieve_center_color(data, contour, labels):
    # get the centroid of the contour
    M = cv2.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    hsv_data = cv2.cvtColor(data, cv2.COLOR_BGR2HSV)
    mean = hsv_data[cY, cX]
    print(mean)
    min_dist = (np.inf, None)
    for label in labels:
        d = np.linalg.norm(label["color"] - np.array(mean))
        if d < min_dist[0]:
            min_dist = (d, label["id"])
    return min_dist[1]


def get_90per_depth(depth_data, cnt):
    """
    Return the 90th percentile depth of the given contour.
    """
    mask = np.zeros(depth_data.shape, dtype="uint8")
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    depth = depth_data[mask == 255]
    print(f"90 tile depth: {np.percentile(depth, 90)}, 50 tile depth: {np.percentile(depth, 50)}, 10 tile depth: {np.percentile(depth, 10)}")
    return np.percentile(depth, 5)


# IK

def get_closest_block(blocks, target):
    min_dist = 100000
    closest_block = None
    # print("blocks", blocks)
    for i in range(blocks.shape[0]):
        block = blocks[i]
        # print("block", block, target)
        x, y, = block[0], block[1]
        dist = np.linalg.norm(np.array([x, y]) - target)
        # print("dist", dist)
        if dist < min_dist:
            min_dist = dist
            closest_block = block.copy()
    return closest_block

def get_traj_grab(pose, orientation):
    """
    return: a list of tuples, each tuple contains a list of joint angles and the duration
    """
    Xw, Yw, Zw = pose
    # Roihn: no longer needs this because we want to align the end effector with the centroid of the block
    # print("previous Xw, Yw", Xw, Yw)
     
    # Xw += OFFSET_GRIPPER * np.sin(np.radians(orientation % 90))
    # Yw += OFFSET_GRIPPER * np.cos(np.radians(orientation % 90))
    # print("current Xw Yw", Xw, Yw)
    trajs = []

    # First go to the top of the block
    q1, q2, q3, q4 = IK_geometric([Xw, Yw, Zw + 70])
    if q1 is not None:
        q5 = np.radians(orientation % 90) + q1
        # print("q1", np.degrees(q1), "q5", np.degrees(q5), "orientation", orientation)
        trajs.append(([q1, q2, q3, q4, q5], 2))
    else:
        return None
    
    # Next go straight downwards
    q1, q2, q3, q4 = IK_geometric([Xw, Yw, Zw + 5])
    if q1 is not None:
        q5 = np.radians(orientation % 90) + q1
        # print("q1", np.degrees(q1), "q5", np.degrees(q5), "orientation", orientation)
        trajs.append(([q1, q2, q3, q4, q5], 2))
    else:
        return None
    
    # Close the gripper
    trajs.append(("grasp", 2))

    # # Lift it a little bit up and rotate the wrist to 0
    q1, q2, q3, q4 = IK_geometric([Xw, Yw, Zw + 100])
    # q5 = q1 - np.pi / 2
    # q5 = np.radians(orientation % 90) + q1
    # print("q1", np.degrees(q1), "q5", np.degrees(q5), "orientation", orientation)
    if q1 is not None:
        q5 = np.radians(orientation % 90) + q1
        trajs.append(([q1, q2, q3, q4, q5], 2))
    else:
        return None
    
    trajs.append((STRAIGHT_UP, 3))

    return trajs

def get_traj_place(pose, straight_orientation=False, orthogonal_orientation=False, is_small=True):
    """
    return: a list of tuples, each tuple contains a list of joint angles and the duration
    """
    Xw, Yw, Zw = pose
    trajs = []
    assert not (straight_orientation and orthogonal_orientation)

    # First go to the top of the position
    q1, q2, q3, q4 = IK_geometric([Xw, Yw, Zw + 150])
    if straight_orientation:
        q5 = q1
    elif orthogonal_orientation:
        q5 = q1 - np.pi / 2
    else:
        q5 = 0
    q5 = clamp(q5)
    if q1 is not None:
        trajs.append(([q1, q2, q3, q4, q5], 2))
    else:
        return None
    # print("q5", np.degrees(q5))
    
    # Next go straight downwards
    if is_small:
        q1, q2, q3, q4 = IK_geometric([Xw, Yw, Zw + 30])
    else:    
        q1, q2, q3, q4 = IK_geometric([Xw, Yw, Zw + 40])
    if straight_orientation:
        q5 = q1
    elif orthogonal_orientation:
        q5 = q1 - np.pi / 2
    else:
        q5 = 0
    q5 = clamp(q5)
    if q1 is not None:
        trajs.append(([q1, q2, q3, q4, q5], 2))
    else:
        return None
    
    # Open the gripper
    trajs.append(("release", 1))

    # Lift it a little bit up and rotate the wrist to 0
    q1, q2, q3, q4 = IK_geometric([Xw, Yw, Zw + 200])
    if straight_orientation:
        q5 = q1
    elif orthogonal_orientation:
        q5 = q1 - np.pi / 2
    else:
        q5 = 0
    q5 = clamp(q5)
    if q1 is not None:
        trajs.append(([q1, q2, q3, q4, q5], 2))
    else:
        return None

    trajs.append((STRAIGHT_UP, 3))
    return trajs

# Competition

## Competition 1: Sort ‘n stack!

def get_shape(cnt, classify_rect=True):
    """
    Return if the given contour is a square or not.
    """
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.07 * peri, True)

    # print(len(approx))
    # shape = "unidentified"
    if len(approx) == 4:
        # Check aspect ratio and possibly angles
        # x, y, w, h = cv2.boundingRect(approx)
        # aspectRatio = float(w) / h
        aspectRatio = get_side_ratio(approx)
        if classify_rect:
            # If we need to tell apart from square and rectangle
            if 0.75 <= aspectRatio <= 1.25:
                # Optionally, further verify by checking angles here.
                # shape = "square (cube)"
                return True
            else:
                # print("not a square")
                # shape = "rectangle"
                return False
        else:
            return True
    # print('not a 4 corner shape')
    return False

def order_points(pts):
    # pts: (4,2)
    # Sort by y-coordinate (top to bottom)
    # If a tie, sort by x-coordinate (left to right)
    pts_sorted = pts[np.lexsort((pts[:,0], pts[:,1]))]
    
    top_left, top_right = pts_sorted[0], pts_sorted[1]
    bottom_left, bottom_right = pts_sorted[2], pts_sorted[3]
    
    # Ensure top_left is actually the leftmost among the top two
    if top_right[0] < top_left[0]:
        top_left, top_right = top_right, top_left
    
    # Ensure bottom_left is actually the leftmost among the bottom two
    if bottom_right[0] < bottom_left[0]:
        bottom_left, bottom_right = bottom_right, bottom_left
    
    return np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

def get_orientation(cnt):
    """
    Return the orientation of the block.
    """
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
    if len(approx) != 4:
        return None, None
    # get the corners
    pts = approx.reshape(4, 2)
    (top_left, top_right, bottom_right, bottom_left) = order_points(pts)
    # print("top_left", top_left)
    # print("top_right", top_right)
    dx = abs(top_right[0] - top_left[0])
    dy = abs(top_right[1] - top_left[1])
    # print(dx, dy)
    # Angle in radians
    angle_radians = np.arctan2(dy, dx)

    # Convert to degrees
    angle_degrees = np.degrees(angle_radians)

    angle_degrees = (angle_degrees // 5) * 5

    corners = np.array([top_left, top_right, bottom_right, bottom_left])
    corners = corners.reshape((-1, 1, 2))
    return angle_degrees, corners


def get_side_ratio(approx):
    """
    Return the orientation of the block.
    """
    pts = approx.reshape(4, 2)
    (top_left, top_right, bottom_right, bottom_left) = order_points(pts)
    # print("top_left", top_left)
    # print("top_right", top_right)
    side_a = np.linalg.norm(top_left - top_right)
    side_b = np.linalg.norm(top_left - bottom_left)
    
    return side_a / side_b

def get_size(cnt):
    """
    Return if the given contour is a small or large block. If large, return True.
    """
    AREA_THRESHOLD = 1100
    return cv2.contourArea(cnt) > AREA_THRESHOLD

def check_clean(world_x, world_y, depth_data, extrinsic_matrix, H):
    """
    Check if the given position and its surronding area is clean.
    """
    print(f"checking world (x,y) = {world_x}, {world_y}")
    x, y, _ = world_to_frame(world_vec=np.array([world_x, world_y, 0, 1]),
                                  extrinsic_matrix=extrinsic_matrix,
                                  use_factory=True,
                                  H=H)
    print(f"frame {x}, {y}, depth = {depth_data[int(y)][int(x)]}")
    # scan the area with a 50x50 grid
    for i in range(-25, 25):
        for j in range(-25, 25):
            depth = depth_data[int(y) + i, int(x) + j]
            if depth < 980:
                return False
    return True
    


## Level 3: Sort and Stack

def sort_blocks_with_size(contours, rgb_image, blocks):
    """
    Filter out the not-square contours, separate the small and large blocks, and sort them by color.
    Return the sorted small and large blocks.
    Each entry in the returned list is a tuple of the block and its color, where block is a list of [x, y, depth, orientation] and color is a string.
    """
    # filter out the not-quare contours
    filtered_contours = []
    filtered_blocks = []
    for contour, block in zip(contours, blocks):
        if get_shape(contour):
            filtered_contours.append(contour)
            filtered_blocks.append(block)
        else:
            print(f"block {block[0]} is filtered out because of shape")

    # separate the small and large blocks
    small_contours, small_blocks = [], []
    large_contours, large_blocks = [], []
    for contour, block in zip(filtered_contours, filtered_blocks):
        if get_size(contour):
            large_contours.append(contour)
            large_blocks.append(block)
        else:
            small_contours.append(contour)
            small_blocks.append(block)

    # sort the small blocks by its color
    # get the color of each block
    sorted_small_blocks = []
    for id, block in enumerate(small_blocks):
        # color = retrieve_area_color(rgb_image, small_contours[id], colors)
        color = retrieve_area_color(rgb_image, small_contours[id], colors_hsv)
        sorted_small_blocks.append((block, color))
    
    # sort the blocks with the sorted id; ascending order
    sorted_small_blocks.sort(key=lambda x: colors_sorted_id[x[1]])
    
    # sort the large blocks by its color
    sorted_large_blocks = []
    for id, block in enumerate(large_blocks):
        # color = retrieve_area_color(rgb_image, large_contours[id], colors)
        color = retrieve_area_color(rgb_image, large_contours[id], colors_hsv)
        sorted_large_blocks.append((block, color))
        
    # sort the blocks with the sorted id; ascending order
    sorted_large_blocks.sort(key=lambda x: colors_sorted_id[x[1]])

    print("small blocks", sorted_small_blocks)
    print("large blocks", sorted_large_blocks)
    
    return sorted_small_blocks, sorted_large_blocks

from collections import defaultdict
import numpy as np

def majority_vote_aggregate(all_detections):
    """
    all_detections: list of (sorted_small_blocks, sorted_large_blocks)
        where each is a list of ( [x,y,depth,orientation], color ).
    
    We do not consider the case when the color is not correctly detected.
    Returns:
        The 'majority' (sorted_small_blocks, sorted_large_blocks) after averaging
        the numeric block fields among all matching detections.
    """

    # Dictionary to store:
    #   key: (tuple_of_small_colors, tuple_of_large_colors)
    #   value: {
    #       "count": how many times this color-sequence pair appears,
    #       "small_blocks": list of the small block-lists for each occurrence,
    #       "large_blocks": list of the large block-lists for each occurrence
    #   }
    detection_dict = defaultdict(lambda: {
        "count": 0,
        "small_blocks": [],
        "large_blocks": []
    })

    # 1. Group detections by color sequence
    for (small_list, large_list) in all_detections:
        # Extract color sequence in order
        small_colors = tuple([pair[1] for pair in small_list])
        large_colors = tuple([pair[1] for pair in large_list])

        key = (small_colors, large_colors)

        detection_dict[key]["count"] += 1
        detection_dict[key]["small_blocks"].append(small_list)
        detection_dict[key]["large_blocks"].append(large_list)

    # 2. Find the key with the largest count (majority)
    majority_key = None
    max_count = 0
    for key, val in detection_dict.items():
        if val["count"] > max_count:
            majority_key = key
            max_count = val["count"]

    if majority_key is None:
        # No detections at all, return empty
        return [], []

    # 3. For the majority key, average the block positions
    # Remember that the 'majority_key' is (small_colors, large_colors).
    # The actual numeric data is in detection_dict[majority_key]["small_blocks"]
    # and detection_dict[majority_key]["large_blocks"].
    val = detection_dict[majority_key]
    small_blocks_list = val["small_blocks"]  # list of sorted_small_blocks from each detection
    large_blocks_list = val["large_blocks"]  # list of sorted_large_blocks from each detection

    # We assume here that all the runs that share the same color sequence
    # have the same length of small_blocks. We'll average position fields index-by-index.
    def average_block_lists(blocks_list):
        """
        blocks_list is a list of lists:
          [
            [ ([x1,y1,d1,o1], color1), ([x2,y2,d2,o2], color2), ... ],
            [ ([x1',y1',d1',o1'], color1'), ([x2',y2',d2',o2'], color2'), ... ],
            ...
          ]
        Returns a single list of ( [avg_x, avg_y, avg_depth, avg_orientation], color )
        """
        # Number of detections for this color-sequence
        n = len(blocks_list)
        # Number of blocks in each detection (assuming consistent across runs)
        m = len(blocks_list[0]) if n > 0 else 0

        averaged_list = []
        for j in range(m):
            # collect positions for block j across all runs
            # also assume the color is the same across these runs for index j
            color = blocks_list[0][j][1]
            xs = []
            ys = []
            depths = []
            orients = []
            for i in range(n):
                x, y, d, o = blocks_list[i][j][0]  # block location
                xs.append(x)
                ys.append(y)
                depths.append(d)
                orients.append(o)

            # average
            mean_x = np.mean(xs)
            mean_y = np.mean(ys)
            mean_d = np.mean(depths)
            mean_o = np.mean(orients)

            averaged_list.append(([mean_x, mean_y, mean_d, mean_o], color))

        return averaged_list

    final_small_blocks = average_block_lists(small_blocks_list)
    final_large_blocks = average_block_lists(large_blocks_list)

    return final_small_blocks, final_large_blocks

