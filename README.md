# Livox Detector - Customized ROS service version 

## Setup
1. Install dependencies (Following dependencies have been tested.).
	- CUDA Toolkit: 10.2
	- python: 3.8
	- numpy: 1.23.1
	- pytorch: 1.8.2
	- ros: melodic
	- rospkg: 1.4.0
	- ros_numpy: 0.0.3 (sudo apt-get install ros-$ros_release-ros-numpy)
	- pyyaml
	- argparse 
2. Build this project
```bash
python3 setup.py develop
```

## Usage
1. Run ROS.
```
roscore
```
2. [USE DETECTOR ALONE]
   Move to 'tools' directory and run test_ros.py (pretrained model: ../pt/livox_model_1.pt or ../pt/livox_model_2.pt).
```
cd tools
python3 test_ros.py --pt ../pt/livox_model_2.pt
```
   Visualize the results.
```
rviz -d rviz.rviz
```
[USE AS LIO_SEGMOT SERVICE] Move to 'tools' directory and run test_ros.py (pretrained model: ../pt/livox_model_1.pt or ../pt/livox_model_2.pt).
```
cd segmot_interface
python3 livox_segmot.py --pt ../pt/livox_model_2.pt
```

3. Play rosbag. (Please adjust the ground plane to 0m and keep it horizontal. The topic of pointcloud2 should be /livox/lidar)
```
rosbag play [bag path]
```
