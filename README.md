# Logistic-object-detection

## Logistic simulation
1.  Clone repository
   ```console
      cd ros2_ws/src
      git clone https://github.com/vidaldani/Logistic-object-detection.git
   ```
2.  Compile workspace
   ```console
      cd ..
      colcon build
      source install/setup.bash 
   ```
3.  Export Turtlebot model
   ```console
      gedit ~/.bashrc
      # copy this instruction at the end: export TURTLEBOT3_MODEL=waffle
      source ~/.bashrc
   ```
4.  Launch logistic simulation:
   ```console
      ros2 launch turtlebot3_gazebo versuchshalle.launch.py
   ```
## Launch object detection
1.  Launch logistic object detection
   ```console
      ros2 launch turtlebot3_recognition launch_yolov8.launch.py
   ```
