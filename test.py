import numpy as np
import cv2
import pyk4a
from pyk4a import Config, PyK4A
from typing import Optional, Tuple
import open3d as o3d

def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img

def get_camera_matrices(k4a):
    """
    获取 Azure Kinect RGB 相机的内参和畸变系数
    """
    calibration = k4a.calibration

    # 获取 color 相机的内参 (3x3) 和畸变系数
    rgb_camera_matrix = np.array(calibration.get_camera_matrix(1))  # 1 = COLOR
    rgb_dist_coeffs = np.array(calibration.get_distortion_coefficients(1))

    return rgb_camera_matrix, rgb_dist_coeffs

def undistort_depth_image(depth_image, camera_matrix, dist_coeffs):
    """
    去畸变深度图像
    """
    h, w = depth_image.shape
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), 1, (w, h)
    )

    # 计算映射表
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1
    )

    #mask = (depth_image > 0).astype(np.uint8) 

    # 使用 OpenCV 的 remap 进行去畸变，采用最近邻插值（cv2.INTER_NEAREST）
    undistorted_depth = cv2.remap(depth_image, map1, map2, interpolation=cv2.INTER_NEAREST)
    #undistorted_depth *= mask
    #x, y, w, h = roi
    #undistorted_depth = undistorted_depth[y:y+h, x:x+w]

    return undistorted_depth

def undistort(depth_image, K, D):

    K = np.array(K).reshape(3, 3)
    D = np.array(D)

    undistorted = cv2.undistort(depth_image, K, D)
    return undistorted

def overlay_images(img1, img2, alpha=0.5):
    """
    叠加两张图像，使用 alpha 混合，alpha 控制透明度
    """
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

def visualize_overlay(color, undistorted, depth_distorted, depth_undistorted):
    """
    直观展示畸变前后的深度图
    """
    color_image = cv2.cvtColor(color, cv2.COLOR_BGRA2BGR)

    # 将深度图归一化到 0-255 并转换为伪彩色图
    undistorted_colormap = cv2.applyColorMap(cv2.convertScaleAbs(undistorted, alpha=255.0/np.max(undistorted)), cv2.COLORMAP_JET)
    depth_distorted_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_distorted, alpha=255.0/np.max(depth_distorted)), cv2.COLORMAP_JET)
    depth_undistorted_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_undistorted, alpha=255.0/np.max(depth_undistorted)), cv2.COLORMAP_JET)
    mask = (depth_undistorted > 0).astype(np.uint8) * 255  # 255 = 有效, 0 = 无效
    mask_colormap = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) # 转换为 3 通道图像
    # 叠加图像
    overlay1 = overlay_images(depth_distorted_colormap, depth_undistorted_colormap, alpha=0.5)
    overlay2 = overlay_images(undistorted_colormap, depth_undistorted_colormap, alpha=0.5)
    #overlay3 = overlay_images(color_image, depth_undistorted_colormap, alpha=0.5)
    overlay3 = overlay_images(color_image, undistorted_colormap, alpha=0.5)
    # 显示结果
    cv2.imshow("Mask (Valid vs Invalid Depth)", mask) # 白色 = 有效, 黑色 = 无效
    cv2.imshow("Mask Colormap", mask_colormap)
    cv2.imshow("UNDistorted", undistorted_colormap)
    cv2.imshow("Distorted Depth", depth_distorted_colormap)
    cv2.imshow("Undistorted Depth", depth_undistorted_colormap)
    cv2.imshow("Overlay1", overlay1)
    cv2.imshow("Overlay2", overlay2)
    cv2.imshow("Overlay3", overlay3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# 启用 color camera 并使用 depth_to_rgb
k4a = PyK4A(Config(
    color_resolution=pyk4a.ColorResolution.RES_1080P,  # 启用 RGB
    depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,  # 深度模式
    synchronized_images_only=True  # 只返回对齐的图像
))
k4a.start()

capture = k4a.get_capture()


# **获取对齐到 RGB 的深度图**
color_image = capture.color
#print(color_image.shape)
cv2.imshow("color", color_image)
depth_image = capture.transformed_depth  # 这里是对齐到 RGB 的深度图
#print(depth_image.shape)
if depth_image is None:
    print("未能获取对齐的深度图，请检查 Kinect 连接")
else:
    print(f"Depth Image Shape: {depth_image.shape}")  # (height, width)
    print(f"Depth Image Min: {np.min(depth_image)}, Max: {np.max(depth_image)}")  # 输出深度图像的最大最小值

rgb_camera_matrix, rgb_dist_coeffs = get_camera_matrices(k4a)
undistorted_depth = undistort_depth_image(depth_image, rgb_camera_matrix, rgb_dist_coeffs)
undistorted = undistort(depth_image, rgb_camera_matrix, rgb_dist_coeffs)

# 显示原始和去畸变深度图
#cv2.imshow("k4a", colorize(depth_image, (None, 5000)))
#cv2.imshow("Undistorted Depth0", colorize(undistorted_depth, (None, 5000)))

visualize_overlay(color_image, undistorted, depth_image, undistorted_depth)

cv2.waitKey(0)
cv2.destroyAllWindows()

k4a.stop()

