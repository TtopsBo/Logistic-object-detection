import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, CalibrationType
from ultralytics import YOLO
import pyk4a
import cv2
import tf_transformations as tf
import transforms3d.quaternions as tq
import time

def create_open3d_obb(center, quaternion, extent, color=(1, 0, 0)):
    """
    将位置 + 四元数 + 尺寸转换成 Open3D 的 OrientedBoundingBox
    """
    obb = o3d.geometry.OrientedBoundingBox()
    obb.center = center
    obb.extent = extent

    # quaternion: (x, y, z, w)
    R = tq.quat2mat([quaternion[3], quaternion[0], quaternion[1], quaternion[2]])  # w, x, y, z
    obb.R = R
    obb.color = color
    return obb

def align_depth_to_color(capture):
    """
    将深度图对齐到彩色相机视角，返回对齐后的深度图（单位：毫米）
    """
    transformation = capture.transformation
    aligned = transformation.depth_image_to_color_camera(capture.depth)
    return aligned

def detect_objects_yolo(detector, image):
    results = detector(image)
    bboxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            bboxes.append([x1, y1, x2, y2])
    return bboxes

def crop_intrinsics(intr, x1, y1):
    intr_crop = o3d.camera.PinholeCameraIntrinsic()
    intr_crop.set_intrinsics(
        width=intr.width - x1,
        height=intr.height - y1,
        fx=intr.get_focal_length()[0],
        fy=intr.get_focal_length()[1],
        cx=intr.get_principal_point()[0] - x1,
        cy=intr.get_principal_point()[1] - y1,
    )
    return intr_crop

def extract_frustum_rgbd(rgb, depth, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    rgb_crop = rgb[y1:y2, x1:x2]
    depth_crop = depth[y1:y2, x1:x2]
    return rgb_crop, depth_crop

def rgbd_to_pointcloud(rgb_crop, depth_crop, intrinsics_crop):
    rgb_crop = np.ascontiguousarray(rgb_crop)
    depth_crop = np.ascontiguousarray(depth_crop)

    color_o3d = o3d.geometry.Image(rgb_crop)
    depth_o3d = o3d.geometry.Image(depth_crop.astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_o3d, depth_o3d,
        depth_scale=1000.0,
        depth_trunc=3.0,
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics_crop)
    return pcd

def apply_cumulative_hist_depth_filter(depth_image, ignore_background = False, resolution = 10, max_height_percent = 1 ):
    # Flatten the depth image to analyze the depth values
    # depth_values = depth_image.flatten()
    # depth_values = depth_values[np.isfinite(depth_values)] # Exclude zero (no data) values
    MIN_DEPTH = 500  # 最小检测距离 (根据模式调整)
    MAX_DEPTH = 3860  # 最大检测距离 (根据模式调整)

    depth_values = depth_image.flatten()
    depth_values = depth_values[(depth_values > 0) & np.isfinite(depth_values)]
    #depth_values = depth_values[(depth_values >= MIN_DEPTH) & (depth_values <= MAX_DEPTH) & np.isfinite(depth_values)]

    if len(depth_values) == 0:
        return depth_image

    # Calculate the histogram of depth values
    bins = round(max(val for val in depth_values if np.isfinite(val)) / resolution)
    hist, bin_edges = np.histogram(depth_values, bins=bins)

    # Find the bin with the maximum count (the most common depth)
    max_peak_index = np.argmax(hist)
    min_height = hist[max_peak_index]*max_height_percent/100
    
    # Detect peaks in the histogram
    threshold = max(np.average(hist[:]), min_height)
    peaks = np.where(hist > threshold)[0]

    min_height = hist[max_peak_index]*max_height_percent/100
    
    if len(peaks) > 0:
        results_dict = []
        # Assume the most significant peak is the object of interest
        for peak in peaks:
            left_edge, right_edge, count = calculate_peak_region(peak, hist, min_height)
            results_dict.append({
            'left_edge': left_edge,
            'right_edge': right_edge,
            'count': count
        })

        # Sort results_dict by 'count' in descending order
        sorted_results = sorted(results_dict, key=lambda x: x['count'], reverse=True)
        
        # Remove duplicates based on (left_edge, right_edge)
        unique_results = { (item['left_edge'], item['right_edge']): item for item in sorted_results }

        # Convert back to a list of dictionaries
        sorted_results = list(unique_results.values())

        # Find the element with the largest count, excluding the last peak if remove_background is True
        if ignore_background:
            dominant_peak = next((elem for elem in sorted_results if elem['right_edge'] != len(hist) - 1), sorted_results[0])
        else:
            dominant_peak = sorted_results[0]

        left_edge = dominant_peak['left_edge']
        right_edge = dominant_peak['right_edge']
        lower_bound = bin_edges[left_edge]
        upper_bound = bin_edges[right_edge]

        # Create a mask to filter out depth values outside the range
        mask = (depth_image >= lower_bound) & (depth_image <= upper_bound)

        # Apply the mask to the depth image
        filtered_image = np.where(mask, depth_image, 0)

        return filtered_image

def calculate_peak_region(peak_index, hist, h_threshold):
    left_edge = peak_index
    right_edge = peak_index

    # Extend edges based on std_dev threshold
    while left_edge > 0 and hist[left_edge] > h_threshold:
        left_edge -= 1
    while right_edge < len(hist) - 1 and hist[right_edge] > h_threshold:
        right_edge += 1

    # Calculate the number of elements within this range
    element_count = np.sum(hist[left_edge:right_edge+1])

    return left_edge, right_edge, element_count

def get_3D_bounding_box(bbox, filtered_image, intrinsics):
    """
    计算点云的3D边界框
    """
    x1, y1, x2, y2 = map(int, bbox[:4])

    depth_values = filtered_image[filtered_image > 0]
    coords = np.column_stack(np.where(filtered_image > 0))


    
    

    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]


    # Open3D 内参格式


    z = depth_values / 1000# Assuming depth is in millimeters
    x = (coords[:, 1] + x1 - cx) * z / fx
    y = (coords[:, 0] + y1 - cy) * z / fy

    # Calculate the 3D bounding box in the world coordinates
    min_x, min_y, min_z = float(np.min(x)), float(np.min(y)), float(np.min(z))
    max_x, max_y, max_z = float(np.max(x)), float(np.max(y)), float(np.max(z))

    # Get the index of the minimum X value
    min_x_index = np.argmin(x)
    z_min_x = z[min_x_index]  # Corresponding Z value for min_x

    # Get the index of the minimum Z value
    min_z_index = np.argmin(z)
    x_min_z = x[min_z_index]  # Corresponding X value for min_z

    # Set the marker's position (center of the bounding box)
    central_x = (min_x + max_x) / 2.0
    central_y = (min_y + max_y) / 2.0
    central_z = (min_z + max_z) / 2.0

    # Create the Pose object for transformation

    position_x = central_x
    position_y = central_y
    position_z = central_z

    quaternion =  calculate_orientation_from_bbox(min_x, z_min_x, x_min_z, min_z)
    orientation_x = quaternion[0]
    orientation_y = quaternion[1]
    orientation_z = quaternion[2]
    orientation_w = quaternion[3]


    size_x = float(max_x - min_x)  # width
    size_y = float(max_y - min_y)  # height
    size_z = float(max_z - min_z)  # depth

    return (position_x, position_y, position_z), (orientation_x, orientation_y, orientation_z, orientation_w), (size_x, size_y, size_z)

def calculate_orientation_from_bbox(min_x, z_min_x, x_min_z, min_z):
    # Calculate the angle (theta) in the XZ plane
    # This is the angle of the line formed between (min_x, z_min_x) and (x_min_z, min_z)
    theta = np.arctan2(min_z - z_min_x, x_min_z - min_x)
    #print(min_x, z_min_x, x_min_z, min_z)
    #print(theta)

    # Convert the angle to a quaternion
    # Since we're working in the XZ plane, we only need to apply a rotation around the Y-axis
    quaternion = tf.quaternion_from_euler(0, theta, 0)
    return quaternion

def get_camera_matrices(k4a):
    """
    获取 Azure Kinect RGB 相机的内参和畸变系数
    """
    calibration = k4a.calibration

    # 获取 color 相机的内参 (3x3) 和畸变系数
    rgb_camera_matrix = np.array(calibration.get_camera_matrix(1))  # 1 = COLOR
    rgb_dist_coeffs = np.array(calibration.get_distortion_coefficients(1))

    return rgb_camera_matrix, rgb_dist_coeffs

def get_cropped_depth(depth_image, bbox):
    """
    根据检测框裁剪深度图像
    """
    x1, y1, x2, y2 = map(int, bbox[:4])
    cropped_depth = depth_image[y1:y2, x1:x2]
    return cropped_depth

def undistort_image(image, camera_matrix, dist_coeffs):
    """
    去畸变深度图像
    """
    h, w = image.shape
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 0, (w, h)
    )

    # 计算映射表
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    #mask = (depth_image > 0).astype(np.uint8) 

    # 使用 OpenCV 的 remap 进行去畸变，采用最近邻插值（cv2.INTER_NEAREST）
    undistorted_image = cv2.remap(image, map1, map2, interpolation=cv2.INTER_NEAREST)

    return undistorted_image
k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        camera_fps=pyk4a.FPS.FPS_5,
        depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
        synchronized_images_only=True,
    )
)



# === Step 1: 初始化 Azure Kinect ===

k4a.start()

while True:
    capture = k4a.get_capture()


    # 检测框格式：x1, y1, x2, y2, conf, class_id
    detector = YOLO(model='src/turtlebot3_recognition/models/yolov11n.pt')

    rgb_camera_matrix, rgb_dist_coeffs = get_camera_matrices(k4a)
    undistorted_depth = undistort_image(capture.transformed_depth, rgb_camera_matrix, rgb_dist_coeffs)


    color_image = capture.color
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(rgb_camera_matrix, rgb_dist_coeffs, (color_image.shape[1], color_image.shape[0]), 0, (color_image.shape[1], color_image.shape[0]))
    color_image = cv2.undistort(color_image, rgb_camera_matrix, rgb_dist_coeffs, None, newCameraMatrix=newcameramtx)
    color = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

    results = detector(color)
    result = results[0]
    if result.boxes is not None and len(result.boxes) > 0:
        print("检测到目标，开始处理")
        break
    else:
        print("未检测到目标，继续采集...")
        time.sleep(0.1)

classes = result.boxes.cls.cpu().numpy()  # 每个框的类别索引
bboxes = result.boxes.xyxy.cpu().numpy() # shape: (N, 6)
color_img = capture.color[..., :3]
colors= color_img[..., ::-1]    # BGR → RGB


raw_pcd = capture.transformed_depth_point_cloud
# valid_mask = np.all(raw_pcd != 0, axis=2)
# points = raw_pcd[valid_mask].astype(np.float32) / 1000.0
# colors = colors[valid_mask] / 255.0

points = raw_pcd.reshape((-1, 3)).astype(np.float32) / 1000.0
colors = colors.reshape((-1, 3)) / 255.0

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
geometries = [pcd]


for bbox, cls in zip(bboxes, classes):
    cropped_depth = get_cropped_depth(undistorted_depth, bbox) 

    filtered_image = apply_cumulative_hist_depth_filter(cropped_depth)
    pos, quat, size = get_3D_bounding_box(bbox, filtered_image, newcameramtx)
    obb = create_open3d_obb(pos, quat, size)
    #pos, quat, size.color(1,0,0)
    print(cls, pos, quat, size)
    geometries.append(obb)

o3d.visualization.draw_geometries(geometries)  


# if result.boxes is None or len(result.boxes) == 0:
#     print("检测到的物体数量为 0")
# else:
#     print("检测到的物体数量为", len(result.boxes))


if k4a.opened:    
    k4a.stop()    
    #k4a.close()
