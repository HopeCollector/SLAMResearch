#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/PointIndices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <iostream>
#include <vector>
#include "seg_rslidar/SegRslidarConfig.h"

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;

ros::Publisher pub_pc_above_ground;
ros::Publisher pub_pc_ground;
ros::Publisher pub_pc_all;
float pass_limit = -1;
float sta_threshold = 1.0;
float planar_threshold = 0.3;


void ros_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    // load cloud
    static pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    static pcl::Indices indices_nan;
    pcl::fromROSMsg(*msg, *cloud);
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices_nan);

    // filter cloud for easier planar segment
    static pcl::IndicesPtr indices_near_ground(new pcl::Indices);
    {
        // pass through filter process
        static pcl::VoxelGrid<PointT> vg;
        static pcl::PassThrough<PointT> pass;
        static bool is_init = false;
        if(!is_init)
        {
            vg.setLeafSize(0.5, 0.5, 0.5);
            vg.setInputCloud(cloud);
            pass.setInputCloud(cloud);
            pass.setFilterFieldName("z");
            is_init = true;
        }

        //vg.filter(*cloud);
        pass.setFilterLimits(-5.0, pass_limit);
        pass.filter(*indices_near_ground);
    }

    // planar segment
    static pcl::PointIndicesPtr indices_ground(new pcl::PointIndices);
    {
        static pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        static pcl::SACSegmentation<PointT> seg;
        static bool is_init = false;
        if(!is_init)
        {
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_PLANE);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setInputCloud(cloud);
            seg.setIndices(indices_near_ground);
            is_init = true;
        }

        seg.setDistanceThreshold(planar_threshold);
        seg.segment(*indices_ground, *coefficients);
    }

    // extract point
    static pcl::PointCloud<PointT>::Ptr cloud_above_ground(new pcl::PointCloud<PointT>);
    static pcl::PointCloud<PointT>::Ptr cloud_ground(new pcl::PointCloud<PointT>);
    {
        static pcl::ExtractIndices<PointT> extr;
        extr.setInputCloud(cloud);
        extr.setIndices(indices_ground);

        extr.setNegative(true);
        extr.filter(*cloud_above_ground);

        extr.setNegative(false);
        extr.filter(*cloud_ground);

        std::vector<bool> is_ground(cloud->size(), false);
        for (auto& idx : indices_ground->indices) is_ground[idx] = true;
        for (size_t i = 0; i < cloud->size(); i++)
        {
            if (is_ground[i])
                cloud->at(i).intensity = 255;
            else
                cloud->at(i).intensity = 0;
        }
    }

    // publish result
    static sensor_msgs::PointCloud2 msg_out_pc_above_ground;
    static sensor_msgs::PointCloud2 msg_out_pc_ground;
    static sensor_msgs::PointCloud2 msg_out_pc_all;
    pcl::toROSMsg(*cloud_above_ground, msg_out_pc_above_ground);
    pub_pc_above_ground.publish(msg_out_pc_above_ground);
    pcl::toROSMsg(*cloud_ground, msg_out_pc_ground);
    pub_pc_ground.publish(msg_out_pc_ground);
    pcl::toROSMsg(*cloud, msg_out_pc_all);
    pub_pc_all.publish(msg_out_pc_all);
    return;
}

void dynamic_callback(seg_rslidar_param::SegRslidarConfig& cfg, uint32_t level)
{
    pass_limit = cfg.pass_limit;
    // sta_threshold = cfg.sta_threshold;
    planar_threshold = cfg.planar_threshold;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ground_removal");
    ros::NodeHandle nh("~");
    dynamic_reconfigure::Server<seg_rslidar_param::SegRslidarConfig> server;

    if (!nh.getParam("pass_limit", pass_limit)) pass_limit = -1.0f;
    if (!nh.getParam("planar_threshold", planar_threshold)) planar_threshold = 0.3f;

    server.setCallback(dynamic_callback);
    auto sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 1, ros_callback);
    pub_pc_all = nh.advertise<sensor_msgs::PointCloud2>("/ground_removal/points_all", 1);
    pub_pc_ground = nh.advertise<sensor_msgs::PointCloud2>("/ground_removal/points_ground", 1);
    pub_pc_above_ground = nh.advertise<sensor_msgs::PointCloud2>("/ground_removal/points_above_ground", 1);

    ros::spin();

    return 0;
}