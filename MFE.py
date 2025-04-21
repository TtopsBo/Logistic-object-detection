import numpy as np
import open3d as o3d
from pyk4a import PyK4A, Config, CalibrationType
from ultralytics import YOLO
import pyk4a
import cv2

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

def apply_cumulative_hist_depth_filter(depth_image, ignore_background = False, resolution = 10, max_height_percent = 5 ):
    # Flatten the depth image to analyze the depth values
    # depth_values = depth_image.flatten()
    # depth_values = depth_values[np.isfinite(depth_values)] # Exclude zero (no data) values
    MIN_DEPTH = 500  # 最小检测距离 (根据模式调整)
    MAX_DEPTH = 3860  # 最大检测距离 (根据模式调整)

    depth_values = depth_image.flatten()
    depth_values = depth_values[(depth_values > 0) & np.isfinite(depth_values)]
    depth_values = depth_values[(depth_values >= MIN_DEPTH) & (depth_values <= MAX_DEPTH) & np.isfinite(depth_values)]

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

# === Step 1: 初始化 Azure Kinect ===
k4a = PyK4A(
    Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
        synchronized_images_only=True
    )
)
k4a.start()

capture = k4a.get_capture()


# 检测框格式：x1, y1, x2, y2, conf, class_id
detector = YOLO(model='src/turtlebot3_recognition/models/yolov11n.pt')

while True:
    color = capture.color  # shape: (H, W, 3)

    color = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)  # Open3D 需要 BGR 格式
    depth = capture.transformed_depth  # 已对齐到 RGB 空间
    #depth = apply_cumulative_hist_depth_filter(depth)
    results = detector(color)
    result = results[0]
    
    if result.boxes is None or len(result.boxes) == 0:
        print("检测到的物体数量为 0")
    else:
        print("检测到的物体数量为", len(result.boxes))
        break

bboxes = result.boxes.xyxy.cpu().numpy() # shape: (N, 6)


# === Step 2: 获取内参（depth_to_rgb）===
calib = k4a.calibration
intr = calib.get_camera_matrix(CalibrationType.COLOR)

fx, fy = intr[0, 0], intr[1, 1]
cx, cy = intr[0, 2], intr[1, 2]
height, width = depth.shape

# Open3D 内参格式
intrinsics = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)

for bbox in bboxes:
    rgb_crop, depth_crop = extract_frustum_rgbd(color, depth, bbox)
    x1, y1, x2, y2 = map(int, bbox[:4])
    intr_crop = crop_intrinsics(intrinsics, x1, y1)

    frustum_pcd = rgbd_to_pointcloud(rgb_crop, depth_crop, intr_crop)
    
    # 法向量计算
    frustum_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    frustum_pcd.orient_normals_consistent_tangent_plane(k=30)



    # Flip to Open3D coordinate system
    frustum_pcd.transform([[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1]])

    # === Step 5: 法向量计算 & RANSAC Manhattan Frame 拟合（同上） ===
    frustum_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    frustum_pcd.orient_normals_consistent_tangent_plane(k=30)

    normals = []
    pcd_copy = frustum_pcd

    for i in range(3):
        plane_model, inliers = pcd_copy.segment_plane(distance_threshold=0.01,
                                                    ransac_n=3,
                                                    num_iterations=1000)
        normal = np.array(plane_model[:3])
        normals.append(normal)
        pcd_copy = pcd_copy.select_by_index(inliers, invert=True)

    # Manhattan Frame
    U, _, _ = np.linalg.svd(np.array(normals).T)
    R = U
    print("Estimated Manhattan Frame:\n", R)

   

        # 计算 frustum 点云的中心
    center = frustum_pcd.get_center()

    # 构建坐标系（轴长可调，默认1.0）
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)

    # 构建变换矩阵：旋转 + 平移
    T = np.eye(4)
    T[:3, :3] = R  # 用你估计出来的曼哈顿框架方向
    T[:3, 3] = center  # 3D 中心点位置

    # 应用变换
    coord_frame.transform(T)

    # 可视化点云 + 坐标系
    o3d.visualization.draw_geometries([frustum_pcd, coord_frame])
if k4a.opened:    
    k4a.stop()    
    k4a.close()
