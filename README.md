# Logistic-object-detection
This project contains a Gazebo simulation with all the [LOCO](https://github.com/tum-fml/loco?tab=readme-ov-file) dataset and the chance to perform object detection using YOLOv8 n with this data.
[Screencast from 08-06-2024 03:02:37 PM.webm](https://github.com/user-attachments/assets/40e422f1-cfc3-4586-b437-fb1ee3a9ccc3)

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
4.  Copy the models inside /turtlebot3_gazebo/models/logistic_objects to ~/.gazebo/models folder
   ```console
   cp -r ~/ros2_ws/src/Logistic-object-detection/src/turtlebot3_gazebo/models/logistic_objects/* ~/.gazebo/models/
   ```
6.  Launch logistic simulation:
   ```console
   ros2 launch turtlebot3_gazebo versuchshalle.launch.py
   ```
## Launch object detection
1.  Install ultralytics (if it is already installed jump to step 2)
   ```console
   pip3 install ultralytics
   ```
2.  On a different terminal launch logistic object detection
   ```console
   cd ros2_ws/
   source install/setup.bash 
   ros2 launch turtlebot3_recognition launch_yolov8.launch.py
   ```
3. Open rviz to visualize the inference results. The configuration file can be found in the folder /turtlebot3_recognition/rviz
```console
rviz2
```
5. In a new terminal launch the keyboard teleoperation to move the robot around and perform object detection
```console
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```
