import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/tumwfml-ubunt6/ros2_ws/src/turtlebot3_ws/install/turtlebot3_teleop'
