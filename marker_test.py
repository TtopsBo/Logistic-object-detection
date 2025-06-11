import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import numpy as np
import json
from IoUtest import get_3D_bounding_box, create_open3d_obb, align_depth_to_color, apply_cumulative_hist_depth_filter, get_camera_matrices, undistort_image
from pyk4a import PyK4A, Config
import pyk4a
from ultralytics import YOLO
import cv2
import tf_transformations as tf
import os
import trimesh
import copy
def o3d_obb_to_trimesh_box(obb: o3d.geometry.OrientedBoundingBox) -> trimesh.Trimesh:
    # 1. 创建 box（中心在原点）
    box = trimesh.creation.box(extents=obb.extent)

    # 2. 构建 4x4 变换矩阵（旋转 + 平移）
    transform = np.eye(4)
    transform[:3, :3] = obb.R
    transform[:3, 3] = obb.center

    # 3. 应用变换
    box.apply_transform(transform)

    return box

def compute_trimesh_iou(pred_box: o3d.geometry.OrientedBoundingBox,
                        gt_box: o3d.geometry.OrientedBoundingBox) -> float:
    mesh1 = o3d_obb_to_trimesh_box(pred_box)
    mesh2 = o3d_obb_to_trimesh_box(gt_box)

    try:
        intersection = mesh1.intersection(mesh2)
        if intersection.is_empty or intersection.volume == 0:
            print("⚠️ Intersection result empty or failed.")
            return 0.0
        inter_vol = intersection.volume
        intersection.export("intersection_debug.ply")
        print("exported")
    except Exception as e:
        print(f"⚠️ Intersection error: {e}")
        return 0.0
    intersection.export("intersection_debug.ply")
    vol1 = mesh1.volume
    vol2 = mesh2.volume
    union = vol1 + vol2 - inter_vol
    return inter_vol / union if union > 0 else 0.0
def build_pointcloud_from_undistorted(rgb_image, depth_image, intrinsics_matrix, depth_scale=1000.0):
    """
    构建去畸变后的彩色点云。
    - rgb_image: BGR 格式图像（已 undistort）
    - depth_image: 深度图（已 undistort），单位 mm
    - intrinsics_matrix: 3x3 内参矩阵（来自 cv2.getOptimalNewCameraMatrix）
    """
    fx, fy = intrinsics_matrix[0, 0], intrinsics_matrix[1, 1]
    cx, cy = intrinsics_matrix[0, 2], intrinsics_matrix[1, 2]
    height, width = rgb_image.shape[:2]

    # 构建 Open3D 内参对象
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    intrinsics.set_intrinsics(width, height, fx, fy, cx, cy)

    # 构建 RGBD 图像
    color_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
    depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))

    rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color=color_o3d,
        depth=depth_o3d,
        depth_scale=depth_scale,
        convert_rgb_to_intensity=False
    )

    # 创建点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsics)

    # 坐标修正（可选：Y/Z 轴翻转）
    # pcd.transform([[1, 0, 0, 0],
    #                [0, -1, 0, 0],
    #                [0, 0, -1, 0],
    #                [0, 0, 0, 1]])

    return pcd

class BoundingBoxEditor:
    def __init__(self, pointcloud, pred_boxes):
        self.pcd = pointcloud

        self.pred_boxes = pred_boxes
        self.gt_boxes =  [copy.deepcopy(box) for box in pred_boxes]
        self.selected_index = 0
        #self.pred_box = pred_boxes[0]
        self.gt_box = self.gt_boxes[0]
        self.app = gui.Application.instance
        self.app.initialize()

        self.window = gui.Application.instance.create_window("3D Box Editor", 1024, 768)
        self.scene = gui.SceneWidget()
        self.scene.scene = rendering.Open3DScene(self.window.renderer)
        self.scene.scene.set_background([1.0, 1.0, 1.0, 1.0])

        # 加载点云和所有框
        self.scene.scene.add_geometry("PointCloud", self.pcd, rendering.MaterialRecord())

        for i, box in enumerate(self.pred_boxes):
            color = [1.0, 0.0, 0.0] if i == 0 else [1.0, 0.0, 0.0]
            mat = rendering.MaterialRecord()
            mat.base_color = [float(c) for c in [1, 0, 0, 1]] #[1.0, 0.0, 0.0, 1.0] 
            mat.shader = "unlitLine"#"defaultUnlit"
            print("Setting color:", mat.base_color)
            self.scene.scene.add_geometry(f"PredBox{i}", box, mat)

        bounds = self.pcd.get_axis_aligned_bounding_box()
        self.scene.setup_camera(60.0, bounds, bounds.get_center())

        self.window.add_child(self.scene)
        self._add_controls()

    def _add_controls(self):
        em = self.window.theme.font_size
        margin = 0.5 * em
        panel = gui.Vert(0.5 * em, gui.Margins(margin))

        # 添加选择框
        self.box_selector = gui.Combobox()
        for i in range(len(self.pred_boxes)):
            self.box_selector.add_item(f"Box {i}")
        self.box_selector.set_on_selection_changed(self._on_box_selected)
        panel.add_child(gui.Label("Select Box to Edit"))
        panel.add_child(self.box_selector)

        self.tx = gui.Slider(gui.Slider.DOUBLE)
        self.ty = gui.Slider(gui.Slider.DOUBLE)
        self.tz = gui.Slider(gui.Slider.DOUBLE)
        for s in (self.tx, self.ty, self.tz):
            s.set_limits(-1.0, 1.0)
            s.set_on_value_changed(self._update_box)
        panel.add_child(gui.Label("Translate X")); panel.add_child(self.tx)
        panel.add_child(gui.Label("Translate Y")); panel.add_child(self.ty)
        panel.add_child(gui.Label("Translate Z")); panel.add_child(self.tz)

        self.rx = gui.Slider(gui.Slider.DOUBLE)
        self.ry = gui.Slider(gui.Slider.DOUBLE)
        self.rz = gui.Slider(gui.Slider.DOUBLE)
        for s in (self.rx, self.ry, self.rz):
            s.set_limits(-180.0, 180.0)
            s.set_on_value_changed(self._update_box)
        panel.add_child(gui.Label("Rotate X (°)")); panel.add_child(self.rx)
        panel.add_child(gui.Label("Rotate Y (°)")); panel.add_child(self.ry)
        panel.add_child(gui.Label("Rotate Z (°)")); panel.add_child(self.rz)

        self.sx = gui.Slider(gui.Slider.DOUBLE)
        self.sy = gui.Slider(gui.Slider.DOUBLE)
        self.sz = gui.Slider(gui.Slider.DOUBLE)
        for s in (self.sx, self.sy, self.sz):
            s.set_limits(0.01, 2.0)
            s.set_on_value_changed(self._update_box)
        panel.add_child(gui.Label("Size X")); panel.add_child(self.sx)
        panel.add_child(gui.Label("Size Y")); panel.add_child(self.sy)
        panel.add_child(gui.Label("Size Z")); panel.add_child(self.sz)

        save_btn = gui.Button("Save bounding box")
        save_btn.set_on_clicked(self._save_box)
        panel.add_child(save_btn)

        iou_btn = gui.Button("Compute IoU")
        iou_btn.set_on_clicked(self._compute_iou)
        panel.add_child(iou_btn)

        self.window.add_child(panel)
        self.window.set_on_layout(lambda ctx: self._on_layout(panel))

    def _on_box_selected(self, name, index):  # 顺序必须是 name, index
        self.selected_index = index
        self.gt_box = self.gt_boxes[index]
        self._load_box_values()

    def _load_box_values(self):
        center = self.gt_box.center
        extent = self.gt_box.extent
        angles = np.degrees(tf.euler_from_matrix(self.gt_box.R))

        self.tx.double_value = center[0]
        self.ty.double_value = center[1]
        self.tz.double_value = center[2]
        self.rx.double_value = angles[0]
        self.ry.double_value = angles[1]
        self.rz.double_value = angles[2]
        self.sx.double_value = extent[0]
        self.sy.double_value = extent[1]
        self.sz.double_value = extent[2]

    def _update_box(self, val):
        center = np.array([self.tx.double_value, self.ty.double_value, self.tz.double_value])
        extent = np.array([self.sx.double_value, self.sy.double_value, self.sz.double_value])
        angles = np.radians([self.rx.double_value, self.ry.double_value, self.rz.double_value])
        R = tf.euler_matrix(*angles, axes='sxyz')[:3, :3]


        new_box = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extent)
        self.gt_boxes[self.selected_index] = new_box
        print("✅ GTBox updated to:", new_box)
        print("修改 GT 之前：")
        print("GT:", self.gt_boxes[0])
        print("Pred:", self.pred_boxes[0])
        self.gt_box = new_box
        self.scene.scene.remove_geometry(f"GTBox{self.selected_index}")

        # 添加新的 GT 框（绿色）
        mat_gt = rendering.MaterialRecord()
        mat_gt.shader = "unlitLine"
        mat_gt.base_color = [0.0, 1.0, 0.0, 1.0]
        mat_gt.line_width = 2.0

        self.scene.scene.add_geometry(f"GTBox{self.selected_index}", new_box, mat_gt)
    def _save_box(self):
        gt_box = self.gt_boxes[self.selected_index]
        data = {
            "center": gt_box.center.tolist(),
            "extent": gt_box.extent.tolist(),
            "rotation": gt_box.R.tolist()
        }
        with open("box_saved.json", "w") as f:
            json.dump(data, f, indent=4)
        print("✅ 已保存 Box 到 box_saved.json")

    def _compute_iou(self):
        if os.path.exists("box_saved.json"):
            with open("box_saved.json", "r") as f:
                data = json.load(f)
                gt_box = o3d.geometry.OrientedBoundingBox(
                    center=np.array(data["center"]),
                    R=np.array(data["rotation"]),
                    extent=np.array(data["extent"])
                )

            pred_box = self.pred_boxes[self.selected_index]
            print("GTBox", gt_box)
            print("PredBox", pred_box)
            iou = compute_trimesh_iou(pred_box, gt_box)
            print(f"📏 几何精确 IoU: {iou:.4f}")
            print(f"IoU with GT: {iou:.3f}")
        else:
            print("⚠️ 没有 box_saved.json，无法计算 IoU")

    def _on_layout(self, panel):
        content_rect = self.window.content_rect
        panel.frame = gui.Rect(content_rect.get_right() - 17 * self.window.theme.font_size,
                               content_rect.y, 15 * self.window.theme.font_size,
                               content_rect.height)
        self.scene.frame = gui.Rect(content_rect.x, content_rect.y,
                                    content_rect.width - panel.frame.width, content_rect.height)

    def run(self):
        self._load_box_values()
        self.app.run()

def compute_3d_iou(box1, box2, voxel_size=0.01):
    min_bound = np.minimum(box1.get_min_bound(), box2.get_min_bound())
    max_bound = np.maximum(box1.get_max_bound(), box2.get_max_bound())

    def voxelize_box(box):
        samples = box.get_box_points()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(samples)
        pcd = pcd.voxel_down_sample(voxel_size)
        coords = np.floor((np.asarray(pcd.points) - min_bound) / voxel_size).astype(int)
        return set(map(tuple, coords))

    vox1 = voxelize_box(box1)
    vox2 = voxelize_box(box2)
    inter = vox1.intersection(vox2)
    union = vox1.union(vox2)
    return len(inter) / len(union) if union else 0

def main():
    # === 初始化相机 ===
    k4a = PyK4A(Config(
        color_resolution=pyk4a.ColorResolution.RES_720P,
        depth_mode=pyk4a.DepthMode.WFOV_2X2BINNED,
        synchronized_images_only=True,
    ))
    k4a.start()

    rgb_camera_matrix, rgb_dist_coeffs = get_camera_matrices(k4a)

    # === 持续采集直到检测成功 ===
    detector = YOLO("src/turtlebot3_recognition/models/yolov11n.pt")
    while True:
        capture = k4a.get_capture()
        color_image = capture.color
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(rgb_camera_matrix, rgb_dist_coeffs, (color_image.shape[1], color_image.shape[0]), 0, (color_image.shape[1], color_image.shape[0]))
        color_image = cv2.undistort(color_image, rgb_camera_matrix, rgb_dist_coeffs, None, newCameraMatrix=newcameramtx)
        aligned_depth = capture.transformed_depth
        undistorted_depth = undistort_image(aligned_depth, rgb_camera_matrix, rgb_dist_coeffs)
        color = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        results = detector(color)
        result = results[0]
        if result.boxes is not None and len(result.boxes) > 0:
            print("检测成功")
            break
        print("未检测到目标，继续循环...")
    pred_boxes = []
    bboxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    raw_pcd = capture.transformed_depth_point_cloud
    # valid_mask = np.all(raw_pcd != 0, axis=2)
    # points = raw_pcd[valid_mask].astype(np.float32) / 1000.0
    # colors = colors[valid_mask] / 255.0
    color_img = capture.color[..., :3]
    colors= color_img[..., ::-1] 
    points = raw_pcd.reshape((-1, 3)).astype(np.float32) / 1000.0
    colors = colors.reshape((-1, 3)) / 255.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    geometries = [pcd]

    
    o3d.io.write_point_cloud("detected_cloud.ply", pcd)

    # === 从检测结果构建预测框 ===
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox[:4])

        # 提取深度子区域并过滤
        cropped = undistorted_depth[y1:y2, x1:x2]
        filtered = apply_cumulative_hist_depth_filter(cropped)

        # 填回完整图（便于 get_3D_bounding_box 使用）


        # 获取 3D 框信息
        pos, quat, size = get_3D_bounding_box(bbox, filtered, newcameramtx)
        # pos = list(pos)
        # pos[1] *= -1
        # pos[2] *= -1
        obb = create_open3d_obb(pos, quat, size, color=(1, 0, 0))
        pred_boxes.append(obb)

    # === 启动 GUI 手动编辑视图 ===
    editor = BoundingBoxEditor(pcd, pred_boxes)
    editor.run()

    
    if k4a.opened:
        k4a.stop()
        #k4a.close()

if __name__ == "__main__":
    main()