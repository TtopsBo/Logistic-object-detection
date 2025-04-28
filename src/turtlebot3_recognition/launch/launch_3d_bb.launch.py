from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.substitutions import LaunchConfiguration,  PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory
from launch.conditions import IfCondition, UnlessCondition

def generate_launch_description():
    # 声明参数
    node_type_arg = DeclareLaunchArgument(
        'node_type',
        default_value='3d',
        description='Type of node to launch: 3d or ekf'
    )

    node_type = LaunchConfiguration('node_type')

    # 包路径
    pkg_dir = get_package_share_directory('turtlebot3_recognition')

    # 根据参数选择不同的子 launch 文件
    launch_3d = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', 'launch_3d_node.launch.py')),
        condition=IfCondition(PythonExpression(["'", node_type, "' == '3d'"]))
    )

    launch_ekf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg_dir, 'launch', 'launch_EKF_node.launch.py')),
        condition=IfCondition(PythonExpression(["'", node_type, "' == 'ekf'"]))
    )

    
    return LaunchDescription([
        node_type_arg,
        launch_3d,
        launch_ekf
    ])
