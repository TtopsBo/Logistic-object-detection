#!/usr/bin/env python3

import rclpy
import numpy as np
from rclpy.node import Node
from rclpy.duration import Duration
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, Vector3
from turtlebot3_recognition.msg import BoundingBox3D  # Replace with your actual package name

import tf2_ros
import tf2_geometry_msgs
import tf_transformations as tf
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException

class BoundingBoxMarkerPublisher(Node):
    def __init__(self):
        super().__init__('bounding_box_marker_publisher')
        
        # Initialize TF2 buffer and listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        self.subscription = self.create_subscription(
            BoundingBox3D,
            '/bounding_boxes_3d',
            self.bounding_box_callback,
            10
        )
        
        self.marker_publisher = self.create_publisher(Marker, '/bounding_box_3d_marker', 10)
        
        self.working_frame = "odom"
        self.registered_objects = []  # List to store registered objects


    def bounding_box_callback(self, msg: BoundingBox3D): 
        class_name = msg.object_name
        frame_id = msg.frame_id
        pose = msg.center
        # Transform the pose to the 'working' frame
        try:
            transform = self.tf_buffer.lookup_transform(
                self.working_frame, 
                frame_id, 
                rclpy.time.Time())  # Use the latest available transform
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, transform)

            # Find existing object ID or generate a new one
            position = np.array([pose.position.x, pose.position.y, pose.position.z])
            marker_id = self.find_existing_object(class_name, position)
            if marker_id is None:
                marker_id = self.generate_marker_id()
                self.registered_objects.append((class_name, marker_id, position))
            
            # Define the marker
            marker = Marker()
            marker.header.frame_id = self.working_frame
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "bounding_box"
            marker.id = marker_id
            marker.text = msg.object_name
            marker.type = Marker.CUBE
            marker.action = Marker.ADD

            # Set the transformed position and orientation
            marker.pose = transformed_pose

            # Set the marker's scale (size of the bounding box)
            marker.scale = msg.size
        
            # Set the color based on the class
            r, g, b = self.get_color_for_class(class_name)
            marker.color.r = r/255.0
            marker.color.g = g/255.0
            marker.color.b = b/255.0
            marker.color.a = msg.conf
            
            # Set marker to disappear after 2 seconds
            marker.lifetime = Duration(seconds=2).to_msg()

            # Publish the marker
            self.marker_publisher.publish(marker)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            # self.get_logger().warn(f"Could not transform marker pose: {e}")
            print("Waiting local frame to become available")
       
        
    def get_color_for_class(self, class_name):
        color_map = {
            "forklift": (255.0, 56.0, 56.0),  # Red
            "pallet_truck": (255.0, 112.0, 31.0),  # Dark Orange
            "pallet": (255.0, 157.0, 151.0),  # Pink
            "small_load_carrier": (255.0, 178.0, 29.0),  # Light Orange
            "stillage": (207.0, 210.0, 49.0),  # Green
        }
        return color_map.get(class_name, (0.5, 0.5, 0.5, 0.5))  # Default: Grey
    
    
    def find_existing_object(self, class_name, position, margin=0.2):
        # Loop through registered objects to find a match within the margin
        for obj in self.registered_objects:
            registered_class_name, registered_id, registered_position = obj
            distance = np.linalg.norm(np.array(position) - np.array(registered_position))
            if registered_class_name == class_name and distance < margin:
                return registered_id
        return None
    
    
    def generate_marker_id(self):
        return len(self.registered_objects) + 1 # Generate a new ID based on list length

class MarkerColor:
    def __init__(self, r, g, b, a):
        self.r = r
        self.g = g
        self.b = b
        self.a = a

    def __iter__(self):
        return iter([self.r, self.g, self.b, self.a])


def main(args=None):
    rclpy.init(args=args)
    node = BoundingBoxMarkerPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
