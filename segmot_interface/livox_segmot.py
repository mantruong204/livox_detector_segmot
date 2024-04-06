#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Some codes are modified from the OpenPCDet.
"""

import os
import glob
import datetime
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

import torch
import torch.nn as nn
from livoxdetection.models.ld_base_v1 import LD_base

import cv2
from cv_bridge import CvBridge

import copy
import rospy
# import ros_numpy
import std_msgs.msg
from geometry_msgs.msg import Point, Quaternion
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField, Image

from visualization_msgs.msg import Marker, MarkerArray
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from lio_segmot.srv import detection, detectionRequest, detectionResponse
from scipy.spatial.transform import Rotation


# from data_process.transformation import lidar_to_camera_box
# from utils.visualization_utils import show_rgb_image_with_boxes



TYPE_INDICES = {'car': 0, 'bus': 1, 'truck': 2, 'pedestrian': 3, 'bimo': 4}

from vis_ros import ROS_MODULE
ros_vis = ROS_MODULE()
last_box_num = 0
last_gtbox_num = 0

marker_array = MarkerArray()
marker_array_text = MarkerArray()

def roty(angle):
    # Rotation about the y-axis.
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def compute_box_3d(dim, location, ry):
    # dim: 3
    # location: 3
    # ry: 1
    # return: 8 x 3
    R = roty(ry)
    h, w, l = dim
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners)
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)


def project_to_image(pts_3d, P):
    # # pts_3d: n x 3
    # # P: 3 x 4
    # # return: n x 2
    pts_3d_homo = np.concatenate([pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]

    return pts_2d.astype(np.int)

    # n = pts_3d.shape[0]
    # pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
    # pts_2d = np.dot(pts_3d_hom, np.transpose(P))  # nx3
    # pts_2d[:, 0] /= pts_2d[:, 2]
    # pts_2d[:, 1] /= pts_2d[:, 2]
    # return pts_2d[:, 0:2]

def draw_box_3d(image, corners, color=(0, 0, 255)):
    ''' Draw 3d bounding box in image
        corners: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7

    '''

    face_idx = [[0, 1, 5, 4],
                [1, 2, 6, 5],
                [2, 3, 7, 6],
                [3, 0, 4, 7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                     (corners[f[(j + 1) % 4], 0], corners[f[(j + 1) % 4], 1]), color, 2, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                     (corners[f[2], 0], corners[f[2], 1]), color, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                     (corners[f[3], 0], corners[f[3], 1]), color, 1, lineType=cv2.LINE_AA)

    return image

def mask_points_out_of_range(pc, pc_range):
    pc_range = np.array(pc_range)
    pc_range[3:6] -= 0.01  #np -> cuda .999999 = 1.0
    mask_x = (pc[:, 0] > pc_range[0]) & (pc[:, 0] < pc_range[3])
    mask_y = (pc[:, 1] > pc_range[1]) & (pc[:, 1] < pc_range[4])
    mask_z = (pc[:, 2] > pc_range[2]) & (pc[:, 2] < pc_range[5])
    mask = mask_x & mask_y & mask_z
    pc = pc[mask]
    return pc

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def boxes_to_corners_3d(boxes3d):
    """
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')

    parser.add_argument('--pt', type=str, default=None, help='checkpoint to start from')

    args = parser.parse_args()
    return args


class ros_demo():
    def __init__(self, model, args=None):
        self.args = args
        self.model = model
        self.starter, self.ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        self.offset_angle = 0
        self.offset_ground = 1.8 
        self.point_cloud_range = [0, -44.8, -2, 224, 44.8, 4]
        
        self.boxes = []
        # Subscribers
        self.image_sub = rospy.Subscriber("/webcam_publisher/camera/image_raw", Image, self.image_callback)

        # Publisher
        self.image_pub = rospy.Publisher("detection/camera/2d_bbox", Image, queue_size=10)
        # Lidar-Camera Calibration matrices
        # self.rotation_matrix = np.array([[-0.99830135, -0.03496315, -0.04660458],
        #                                  [0.04760497, -0.02834738, -0.99846392],
        #                                  [0.03358832, -0.99898649, 0.02996365]])

        # self.translation_offsets = np.array([-0.00091186, -0.09075158, -0.06664682])

        # self.euler_angles = np.array([-1.5408112700226857, -0.03359464033115832, 3.0939427751598942])
        
        self.lidar_cam_proj_mat = np.array([[-0.99830135, -0.03496315, -0.04660458, -0.00091186],
                                         [0.04760497, -0.02834738, -0.99846392, -0.09075158],
                                         [0.03358832, -0.99898649, 0.02996365, -0.06664682]])

        # Camera matrix
        self.camera_matrix = np.array([[497.77969, 0, 290.50365],
                                        [0, 497.39897, 236.15821],
                                        [0, 0, 1]])
        self.distortion_coefficients = np.array([0.006494, -0.026018, -0.000978, -0.001871, 0.])
        
        self.projection_matrix = np.array([[496.40695, 0., 288.95222, 0.],
                                  [0., 497.66645, 235.77096, 0.],
                                  [0., 0., 1., 0.]])

        self.bridge = CvBridge()

    # def receive_from_ros(self, msg):
    #     pc = ros_numpy.numpify(msg)
    #     points_list = np.zeros((pc.shape[0], 4))
    #     points_list[:, 0] = copy.deepcopy(np.float32(pc['x']))
    #     points_list[:, 1] = copy.deepcopy(np.float32(pc['y']))
    #     points_list[:, 2] = copy.deepcopy(np.float32(pc['z']))
    #     points_list[:, 3] = copy.deepcopy(np.float32(pc['intensity']))

    #     # preprocess 
    #     points_list[:, 2] += points_list[:, 0] * np.tan(self.offset_angle / 180. * np.pi) + self.offset_ground
    #     rviz_points = copy.deepcopy(points_list)
    #     points_list = mask_points_out_of_range(points_list, self.point_cloud_range)

    #     input_dict = {
    #             'points': points_list,
    #             'points_rviz': rviz_points
    #             }

    #     data_dict = input_dict
    #     return data_dict

    def image_callback(self, data):
        try:
            # Convert ROS Image message to OpenCV image
            cv_image_raw = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except Exception as e:
            rospy.logerr(e)
            return

        # Rectify the image using the camera matrix
        cv_image_rect = cv2.undistort(cv_image_raw, self.camera_matrix, None)

        bbox_3d_list = self.boxes

        # # Transform and draw 2D bounding boxes for each set of 3D bounding boxes
        img_rect_with_boxes = cv_image_rect.copy()  # Initialize the image with 2D bounding boxes
        # for bbox_3d in bbox_3d_list:
        #     img_rect_with_boxes = self.transform_and_draw(img_rect_with_boxes, bbox_3d)
        
        

        if len(bbox_3d_list) > 0:
            for bbox_3d in bbox_3d_list:
                location, dim, ry = bbox_3d[0:3], bbox_3d[3:6], bbox_3d[6]
                if location[1] > -2.0:  # The object is too close to the camera, ignore it during visualization
                    continue
                corners_3d = compute_box_3d(dim, location, ry)
                corners_2d = project_to_image(corners_3d, self.projection_matrix)
                print(f"corners_3d = {corners_3d}\n corners_2d = {corners_2d}")
                img_rect_with_boxes = draw_box_3d(img_rect_with_boxes, corners_2d, (0,255,255))
        # Convert the image back to ROS Image message
        image_msg = self.bridge.cv2_to_imgmsg(img_rect_with_boxes, "bgr8")

        # Publish the image with 2D bounding boxes
        self.image_pub.publish(image_msg)


    
    @staticmethod
    def load_data_to_gpu(batch_dict):
        for key, val in batch_dict.items():
            if not isinstance(val, np.ndarray):
                continue
            else:
                batch_dict[key] = torch.from_numpy(val).float().cuda()

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            if key in ['points']:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                    coors.append(coor_pad)
                ret[key] = np.concatenate(coors, axis=0)
        ret['batch_size'] = batch_size
        return ret

    def online_inference(self, request: detectionRequest):
        header = request.cloud.header

        response = detectionResponse()
        response.detections = BoundingBoxArray()
        response.detections.header = header

        points_list = []
        for point in pcl2.read_points(request.cloud, skip_nans=True, field_names=("x", "y", "z", "intensity")):
            if point[0] == 0 and point[1] == 0 and point[2] == 0:
                continue
            if np.abs(point[0]) < 2.0 and np.abs(point[1]) < 1.5:
                continue
            points_list.append(point)

        if len(points_list) == 0:
            return response
        points_list = np.asarray(points_list)


        points_list[:, 2] += points_list[:, 0] * np.tan(self.offset_angle / 180. * np.pi) + self.offset_ground
        rviz_points = copy.deepcopy(points_list)
        points_list = mask_points_out_of_range(points_list, self.point_cloud_range)

        data_dict = {
                'points': points_list,
                'points_rviz': rviz_points
                }
        # print(f"data_dict = {data_dict}")
        data_infer = ros_demo.collate_batch([data_dict])
        ros_demo.load_data_to_gpu(data_infer)
        
        self.model.eval()
        with torch.no_grad(): 
            torch.cuda.synchronize()
            self.starter.record()
            pred_dicts = self.model.forward(data_infer)
            self.ender.record()
            torch.cuda.synchronize()
            curr_latency = self.starter.elapsed_time(self.ender)
        print('det_time(ms): ', curr_latency)
        
        data_infer, pred_dicts = ROS_MODULE.gpu2cpu(data_infer, pred_dicts)
       
        

        # print(f"pred_dicts = {pred_dicts}")
        boxes = pred_dicts[0]['pred_boxes']
        # each box has [x, y, z, dx, dy, dz, heading]
        self.boxes = boxes
        # self.boxes = boxes_to_corners_3d(boxes)
        # print(f"boxes = {boxes}")

        print(f"object num. = {len(boxes)}\n________________________________________")     # debugggggg
        
        marker_array.markers.clear()
        marker_array_text.markers.clear()

        for obid in range(len(boxes)):
            ob = boxes[obid]
            
            box = BoundingBox()
            box.header = header
            # box.label = TYPE_INDICES[pred_dicts[0]['pred_labels'][obid]]

            box.pose.position.x = ob[0]
            box.pose.position.y = ob[1]
            box.pose.position.z = ob[2] - 1.8
            # print(
            #     "Position:" + 
            #     "\n x = " + str(box.pose.position.x) +
            #     "\n y = " + str(box.pose.position.y) +
            #     "\n z = " + str(box.pose.position.z) 
            # )

            box.dimensions.x = ob[3]
            box.dimensions.y = ob[4]
            box.dimensions.z = ob[5]
            # print(
            #     "Dimension:"+
            #     "\n x = " + str(box.dimensions.x) +
            #     "\n y = " + str(box.dimensions.y) +
            #     "\n z = " + str(box.dimensions.z) 
            # )
            
            heading_angle = ob[6]             

            c = np.cos(heading_angle)
            s = np.sin(heading_angle)
            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            quat = Rotation.from_matrix(rot).as_quat()  # (x, y, z, w)

            box.pose.orientation.x = quat[0]
            box.pose.orientation.y = quat[1]
            box.pose.orientation.z = quat[2]
            box.pose.orientation.w = quat[3]

            # print(
            #     "Quaternion:"+
            #     "\n x = " + str(box.pose.orientation.x) +
            #     "\n y = " + str(box.pose.orientation.y) +
            #     "\n z = " + str(box.pose.orientation.z) +
            #     "\n w = " + str(box.pose.orientation.w) +
            #     "\n____________________________________"
            # )

            response.detections.boxes.append(box)

        return response

        

if __name__ == '__main__':
    args = parse_config()
    model = LD_base()

    checkpoint = torch.load(args.pt, map_location=torch.device('cpu'))  
    model.load_state_dict({k.replace('module.', ''):v for k, v in checkpoint['model_state_dict'].items()})
    model.cuda()

    demo_ros = ros_demo(model, args)
    
    # sub = rospy.Subscriber(
    #     "/velodyne_points", PointCloud2, queue_size=10, callback=demo_ros.online_inference)
    rospy.Service('lio_segmot_detector', detection, demo_ros.online_inference)
    print("Set up SERVICE!!!")
    
    

    rospy.spin()
    
