#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Image.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/range_image/range_image.h>
#include <pcl_conversions/pcl_conversions.h>
#include <dynamic_reconfigure/server.h>

#include "range_img/RangeImgConfig.h"

using Point = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<Point>;

pcl::RangeImage::Ptr rangeimg_ptr(new pcl::RangeImage);
float angular_resolution = pcl::deg2rad(1.0f);
float max_angle_width = pcl::deg2rad(360.0f);
float max_angle_height = pcl::deg2rad(180.0f);
Eigen::Affine3f sensorPose = static_cast<Eigen::Affine3f>(Eigen::Translation3f(0.0f,0.0f,0.0f));

ros::Publisher pub_rangeimg;

pcl::visualization::RangeImageVisualizer range_viewer("Range Image");

void ros_callback(const sensor_msgs::PointCloud2ConstPtr msg)
{
    static PointCloud pointCloud;
    static sensor_msgs::PointCloud2 out_rangeimg;

    pcl::fromROSMsg(*msg, pointCloud);
    rangeimg_ptr->createFromPointCloud(pointCloud, angular_resolution,
                                       max_angle_width, max_angle_height, sensorPose,
                                       pcl::RangeImage::LASER_FRAME,
                                       0.0f, 0.0f, 1);
    range_viewer.showRangeImage(*rangeimg_ptr);
    pcl::toROSMsg(*rangeimg_ptr, out_rangeimg);

    pub_rangeimg.publish(out_rangeimg);
}

void dynamic_callback(range_image_viewer::RangeImgConfig &cfg, uint32_t level)
{
    angular_resolution = pcl::deg2rad(cfg.angular_resolution);
    max_angle_width = pcl::deg2rad(cfg.max_angle_width);
    max_angle_height = pcl::deg2rad(cfg.max_angle_height);
    sensorPose = static_cast<Eigen::Affine3f>(Eigen::Translation3f(
        cfg.sensor_pos_x, cfg.sensor_pos_y, cfg.sensor_pos_z
    ));
}

int main (int argc, char** argv)
{
    ros::init(argc, argv, "range_img_node");
    ros::NodeHandle nh("~");

    dynamic_reconfigure::Server <range_image_viewer::RangeImgConfig> server;
    server.setCallback(dynamic_callback);

    auto sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 2, ros_callback);
    pub_rangeimg = nh.advertise<sensor_msgs::PointCloud2>("/rslidar_rangeimg", 1);

    ros::Rate r(30);
    while(ros::ok() && !range_viewer.wasStopped())
    {
        ros::spinOnce();
        range_viewer.spinOnce();
        r.sleep();
    }

    return 0;
}