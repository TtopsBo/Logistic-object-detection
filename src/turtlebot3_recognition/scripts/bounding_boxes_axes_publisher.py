import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.duration import Duration
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Vector3, TransformStamped
from turtlebot3_recognition.msg import BoundingBox3D  # Replace with你的msg包
import tf2_ros
import tf2_geometry_msgs
import tf_transformations as tf
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class BoundingBoxTFPublisher(Node):
    def __init__(self):
        super().__init__('bounding_box_tf_publisher')
        
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)  # **TF 广播器**
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.subscription = self.create_subscription(
            BoundingBox3D,
            '/bounding_boxes_3d',
            self.bounding_box_callback,
            10
        )
        
        self.working_frame = "camera_color_optical_frame"  
        

    def bounding_box_callback(self, msg: BoundingBox3D): 
        class_name = msg.object_name
        frame_id = msg.frame_id  # 物体的原始坐标系，比如 "camera_color_optical_frame"
        pose = msg.center

        # **创建 TF 变换**
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.working_frame  
        transform.child_frame_id = f"{class_name}_bbox"  

        # **设置 TF 位置**
        transform.transform.translation.x = pose.position.x
        transform.transform.translation.y = pose.position.y
        transform.transform.translation.z = pose.position.z

        # **转换 Orientation 从 optical 到 camera_link**
        optical_to_base_quat = [0.5, -0.5, 0.5, -0.5]  # 从 optical 到 camera_link 的四元数
        bbox_quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        corrected_quat = tf.quaternion_multiply(bbox_quat, optical_to_base_quat)

        transform.transform.rotation.x = corrected_quat[0]
        transform.transform.rotation.y = corrected_quat[1]
        transform.transform.rotation.z = corrected_quat[2]
        transform.transform.rotation.w = corrected_quat[3]

        # **广播 TF**
        self.tf_broadcaster.sendTransform(transform)
        #self.get_logger().info(f"Published TF for {class_name} at {pose.position.x}, {pose.position.y}, {pose.position.z}")


def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxTFPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
