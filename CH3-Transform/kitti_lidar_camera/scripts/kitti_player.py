#!/usr/bin/env python

import _thread
import os
import sys

sys.path.insert(0, '/home/lawskiy/Documents/LearnSLAM/rslidar_ws/src/kitti_lidar_camera/scripts')

import rospy
import std_msgs.msg
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
# ground truth bounding box
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
import tf
from tf import transformations as trans

import cv2
import datetime as dt
import numpy as np

import pykitti.utils as kitti

import utils.KittiPreprocessor as Preprocessor
import utils.Publisher as KittiPublisher

'''
    sudo apt-get install python-evdev
'''
from evdev import InputDevice
from select import select

KEY_IDLE=0
KEY_SPACE=57
KEY_LEFT=105
KEY_RIGHT=106
NEXT_FRAME=KEY_RIGHT
LAST_FRAME=KEY_LEFT
KEY_VAL=KEY_IDLE
def on_keyboard(name_dev):
    global KEY_VAL
    dev = InputDevice(name_dev)
    while True:
        select([dev], [], [])
        for event in dev.read():
            if (event.value!=0) and (event.code!=0):
                # KEY_VAL will keep until next pressed
                KEY_VAL = event.code


if __name__ == "__main__":
    # ROS parameters
    keyboard_file = rospy.get_param("/kitti_player/keyboard_file", "/dev/input/event3")

    vel_frame_ = rospy.get_param("/kitti_player/vel_frame", "velodyne")
    imu_frame_ = rospy.get_param("/kitti_player/imu_frame", "imu")
    world_frame_ = rospy.get_param("/kitti_player/world_frame", "world")

    mode = rospy.get_param("/kitti_player/mode", "observation")
    fps = rospy.get_param("/kitti_player/fps", 10)
    filter_by_camera_angle_ = rospy.get_param("/kitti_player/filter_by_camera_angle", True)
    care_objects = rospy.get_param("/kitti_player/care_objects", ['Car','Van','Truck','Pedestrian','Sitter','Cyclist','Tram','Misc'])
    path = rospy.get_param("/kitti_player/kitti_data_path", "/home/lawskiy/Documents/LearnSLAM/rslidar_ws/data/2011_09_26/2011_09_26_drive_0001_sync")

    playing = False
    # open a keyboard listen thread on play mode
    if mode == "play":
        try:
            _thread.start_new_thread(on_keyboard, (keyboard_file,))
        except Exception as e:
            print(str(e))
            print("Error: unable to start keyboard listen thread.")

    rospy.init_node("kitti_player")
    # Publisher of Kitti raw data: point cloud & image & ground truth
    pub_points = rospy.Publisher("/kitti/points_raw", PointCloud2, queue_size=1000000)
    pub_points_rgb = rospy.Publisher("/kitti/points_rgb", PointCloud2, queue_size=1000000)
    pub_img = rospy.Publisher("/kitti/img_raw", Image, queue_size=1000000)
    pub_img_depth = rospy.Publisher("/kitti/img_depth", Image, queue_size=1000000)
    ground_truth_pub_ = rospy.Publisher("/kitti/bb_raw", PoseArray, queue_size=1000000)

    object_marker_pub_ = rospy.Publisher("/kitti/bb_marker", MarkerArray, queue_size=1000000)

    # Publisher of bounding box corner vertex
    pub_vertex = rospy.Publisher("/kitti/points_corners", PointCloud2, queue_size=1000000)
    # pub_img_bb = rospy.Publisher("/kitti/objects_bb", PointCloud2, queue_size=1000000)
    # Publisher of bounding box
    pub_clusters = rospy.Publisher("/kitti/points_clusters", PointCloud2, queue_size=1000000)

    static_tf_sender = tf.TransformBroadcaster()
    pose_tf_sender = tf.TransformBroadcaster()

    # Shared header for synchronization
    header_ = std_msgs.msg.Header()
    header_.stamp = rospy.Time.now()
    header_.frame_id = "velodyne"

    fps = rospy.Rate(fps)

    timestamp_file = path + "/" + "velodyne_points/timestamps.txt"

    pcd_path = None
    bin_path = path + "/velodyne_points/data"
    oxts_path = path + "/oxts/data"
    # img_path = path + "/" + "image_0[0-3]/data/"
    img_path = path + "/image_02/data"

    calib_cam_to_cam_file = path + "/../calib_cam_to_cam.txt"
    calib_velo_to_cam_file = path + "/../calib_velo_to_cam.txt"
    calib_imu_to_velo_file = path + "/../calib_imu_to_velo.txt"

    tracklet_file = path + "/" + "tracklet_labels.xml"
    use_gt = os.path.exists(tracklet_file)


    ######################## 读取文件 ###########################
    """
    读取文件
    - 时间辍
    - 点云文件
    - GPS/IMU 文件
    - 2号彩色相机的图像文件
    - 激光雷达到相机的坐标系变换文件
    """
    timestamps = []
    with open(timestamp_file, 'r') as f:
        for line in f.readlines():
            # NB: datetime only supports microseconds, but KITTI timestamps
            # give nanoseconds, so need to truncate last 4 characters to
            # get rid of \n (counts as 1) and extra 3 digits
            t = dt.datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
            timestamps.append(t)

    bin_files = []
    if os.path.isdir(bin_path):
        for f in os.listdir(bin_path):
            if os.path.isdir(f):
                continue
            else:
                bin_files.append(f)
    bin_files.sort()

    pose_files = []
    if os.path.isdir(oxts_path):
        for f in os.listdir(oxts_path):
            if os.path.isdir(f):
                continue
            else:
                pose_files.append(oxts_path + "/" + f)
    pose_files.sort()
    poses = []
    for pose in kitti.get_oxts_packets_and_poses(pose_files):
        poses.append(pose[1])

    img_files = []
    if os.path.isdir(img_path):
        for f in os.listdir(img_path):
            if os.path.isdir(f):
                continue
            else:
                img_files.append(f)
    img_files.sort()

    calib_imu2velo = kitti.read_calib_file(calib_imu_to_velo_file)
    imu2vel = np.zeros((4, 4))
    imu2vel[:3,:3] = np.array(calib_imu2velo['R']).reshape(-1, 3)
    imu2vel[:3,3] = calib_imu2velo['T']
    imu2vel[3,3] = 1.

    vel2imu = trans.inverse_matrix(imu2vel)
    translation_static = trans.translation_from_matrix(vel2imu)
    quaternion_static = trans.quaternion_from_matrix(vel2imu)

    calib_cam2cam = kitti.read_calib_file(calib_cam_to_cam_file)
    # To project a 3D point x in reference camera coordinates to a point y on the i'th image plane,
    # the rectifying rotation matrix of the reference camera: R_rect_00 must be considered as well.
    R_rect_00 = np.zeros((4, 4))
    R_rect_00[:3,:3] = np.array(calib_cam2cam['R_rect_00']).reshape(-1, 3)
    R_rect_00[3,3] = 1.
    print("     ----------- R_rect_00 -----------")
    print(R_rect_00)
    # To project to a point in the i'th camera image, 0...3
    P_rect_02 = np.zeros((3, 4))
    P_rect_02 = np.array(calib_cam2cam['P_rect_02']).reshape(-1, 4)
    print("     ----------- P_rect_02 -----------")
    print(P_rect_02)

    calib_velo2cam = kitti.read_calib_file(calib_velo_to_cam_file)
    vel2cam0 = np.zeros((4, 4))
    vel2cam0[:3,:3] = np.array(calib_velo2cam['R']).reshape(-1, 3)
    vel2cam0[:3,3] = calib_velo2cam['T']
    vel2cam0[3,3] = 1.
    print("     ----------- T_velo_cam -----------")
    print(vel2cam0)

    # compute transform matrix
    T_velo_to_cam = np.dot(R_rect_00, vel2cam0)
    print("     ----------- T_velo_cam -----------")
    print(T_velo_to_cam)
    print("\n\n")
    # P_velo_to_img = np.dot(P_rect_02, T_velo_to_cam)
    P_velo_to_img = np.dot(P_rect_02, np.dot(R_rect_00, vel2cam0))
    print("     ----------- P_velo_image -----------")
    print(P_velo_to_img)
    print("\n\n")
    ######################## 读取文件 ###########################

    # bounding_boxes[frame index]
    if use_gt:
        bounding_boxes, tracklet_counter = Preprocessor.read_label_from_xml(tracklet_file, care_objects)

    idx = 0
    # support circular access ...-2,-1,0,1,2...
    while idx < len(bin_files):
        # CTRL+C exit
        if rospy.is_shutdown():
            print("")
            print("###########")
            print("[INFO] ros node had shutdown...")
            sys.exit(0)

        ##TODO read data 加载点云数据
        pc = Preprocessor.load_pc_from_bin(bin_path + "/" + bin_files[idx])

        print("\n[",timestamps[idx],"]","# of Point Clouds:", pc.size)

        # 加载图像数据
        image = cv2.imread(img_path + "/" + img_files[idx])

        ##TODO timestamp 根据点云的时间辍数据生成 ROS 时间辍
        #header_.stamp = rospy.Time.from_sec(timestamps[idx].total_seconds())
        # print (timestamps[idx] - dt.datetime(1970,1,1)).total_seconds()
        header_.stamp = rospy.Time.from_sec((timestamps[idx] - dt.datetime(1970,1,1)).total_seconds())

        # 维护一个 tf 坐标变换关系
        """
            :param translation: the translation of the transformtion as a tuple (x, y, z)
            :param rotation: the rotation of the transformation as a tuple (x, y, z, w)
            :param time: the time of the transformation, as a rospy.Time()
            :param child: child frame in tf, string
            :param parent: parent frame in tf, string
            Broadcast the transformation from tf frame child to parent on ROS topic ``"/tf"``.
            
            world - imu - vel
        """
        static_tf_sender.sendTransform(translation_static, quaternion_static,
                                       header_.stamp,
                                       vel_frame_, imu_frame_)

        translation = trans.translation_from_matrix(poses[idx])
        quaternion = trans.quaternion_from_matrix(poses[idx])
        pose_tf_sender.sendTransform(translation, quaternion,
                                     header_.stamp,
                                     imu_frame_, world_frame_)
        if mode != "play":
            img_window = "Kitti"
            # Image Window Setting
            screen_res = 1280, 720
            scale_width = screen_res[0] / image.shape[1]
            scale_height = screen_res[1] / image.shape[0]
            scale = min(scale_width, scale_height)
            window_width = int(image.shape[1] * scale)
            window_height = int(image.shape[0] * scale)*2
            cv2.namedWindow(img_window, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(img_window, window_width, window_height)

        if filter_by_camera_angle_:
            # Camera angle filters
            # 返回以 x 轴为中心，左右两边各 45度 共 90度 的视角内的点云，将多余的点云扔掉
            pc = Preprocessor.filter_by_camera_angle(pc)
            # XYZRGB point cloud and depth image
            """
            首先，将点云投影到相机平面，得到每个点在相机平面内的像素坐标
            然后
            - 将每个点对应像素点位置的颜色复制到 rgb 点云中
            - 将深度图中每个像素的颜色按照对应点云（坐标系已转换到相机坐标系 cam2）中的点到原点的距离进行染色
            最后返回 pc_rgb image_depth

            投影中使用 pos_img = P_rect_02 * T_vel_to_cam * pos_xyz
            转换坐标系中使用 pos_xyz_cam = T_vel_to_cam * pos_xyz_vel

            点云中每个点的坐标系转换流程，两个变换矩阵负责的变换范围
            vel - cam0 - cam2(基于 cam2 的坐标系) - cam2_img(此时的三维点被投影到相机平面)
            |________T_vel_to_cam_____________|   |_________P_vel_to_cam__________|
            """
            pc_rgb, image_depth = Preprocessor.lidar_camera_fusion(pc, image, T_velo_to_cam, P_velo_to_img)

        places = None
        rotates_z = None
        size = None
        corners = None
        if use_gt and idx in list(bounding_boxes.keys()):
            places = bounding_boxes[idx]["place"]
            # avoid IndexError: too many indices for array
            if bounding_boxes[idx]["rotate"].ndim > 1:
                rotates_z = bounding_boxes[idx]["rotate"][:, 2]
            else:
                rotates_z = bounding_boxes[idx]["rotate"][2]
            size = bounding_boxes[idx]["size"]

            # Create 8 corners of bounding box
            corners = Preprocessor.get_boxcorners(places, rotates_z, size)

        KittiPublisher.publish_raw_clouds(pub_points, header_, pc)
        KittiPublisher.publish_rgb_clouds(pub_points_rgb, header_, pc_rgb)

        if corners is not None:
            KittiPublisher.publish_ground_truth_boxes(ground_truth_pub_, header_, places, rotates_z, size)
            # publish_bounding_vertex(pub_vertex, header_, corners.reshape(-1, 3))
            # publish_img_bb(pub_img_bb, header_, corners.reshape(-1, 3))
            KittiPublisher.publish_ground_truth_markers(object_marker_pub_, header_, corners.reshape(-1, 3))
            # publish_clusters(pub_clusters, header_, pc, corners.reshape(-1, 3))
        elif use_gt:
            print("no object in current frame: " + bin_files[idx])
            # publish empty message
            KittiPublisher.publish_ground_truth_boxes(ground_truth_pub_, header_, None, None, None)
            KittiPublisher.publish_ground_truth_markers(object_marker_pub_, header_, None)


        """
            publish RGB image
        """
        KittiPublisher.publish_raw_image(pub_img_depth, header_, image_depth)
        KittiPublisher.publish_raw_image(pub_img, header_, image)
        print("###########")
        print("[INFO] Show image: ",img_files[idx])
        if mode != "play":
            cv2.imshow(img_window, image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            idx += 1
        else:
            fps.sleep()
            # Keyboard control logic
            if playing:
                if KEY_VAL==KEY_SPACE:
                    playing = False
                    KEY_VAL = KEY_IDLE
                else:
                    idx += 1
            while not playing:
                if KEY_VAL==NEXT_FRAME:
                    idx += 1
                    if idx >= len(bin_files):
                        idx = 0
                    KEY_VAL = KEY_IDLE
                    break
                elif KEY_VAL==LAST_FRAME:
                    idx -= 1
                    # if idx < 0:
                    #     idx = 0
                    KEY_VAL = KEY_IDLE
                    break
                elif KEY_VAL==KEY_SPACE:
                    playing = True
                    idx += 1
                    KEY_VAL = KEY_IDLE
                    break
                else:
                    # CTRL+C exit
                    if rospy.is_shutdown():
                        print("")
                        print("###########")
                        print("[INFO] ros node had shutdown...")
                        sys.exit(-1)

    print("###########")
    print("[INFO] All data played...")