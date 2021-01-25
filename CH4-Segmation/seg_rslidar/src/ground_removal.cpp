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

using PointType = pcl::PointXYZ;

ros::Publisher pub_pc_above_ground;
ros::Publisher pub_pc_ground;
float pass_limit = -1;
float sta_threshold = 1.0;
float planar_threshold = 0.3;


void ros_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    // load cloud
    static pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    static pcl::Indices indices_nan;
    pcl::fromROSMsg(*msg, *cloud);
    pcl::removeNaNFromPointCloud(*cloud, *cloud, indices_nan);

    // filter cloud for easier planar segment
    static pcl::IndicesPtr indices_near_ground(new pcl::Indices);
    {
        // pass through filter process
        static pcl::PassThrough<PointType> pass;
        pass.setInputCloud(cloud);
        pass.setFilterFieldName("z");
        pass.setFilterLimits(-5.0, pass_limit);
        pass.filter(*indices_near_ground);

        // static filter
        // 点太稀疏了，就像老子的头发，不过滤了，心理难受😢
        // static pcl::StatisticalOutlierRemoval<PointType> sor;
        // sor.setInputCloud(cloud);
        // sor.setIndices(indices_near_ground);
        // sor.setMeanK(50);
        // sor.setStddevMulThresh(sta_threshold); // this can be configed
        // sor.filter(*indices_near_ground);
    }

    // planar segment
    static pcl::PointIndicesPtr indices_ground(new pcl::PointIndices);
    {
        static pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        static pcl::SACSegmentation<PointType> seg;
        seg.setOptimizeCoefficients(true);
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(planar_threshold);
        seg.setInputCloud(cloud);
        seg.setIndices(indices_near_ground);
        seg.segment(*indices_ground, *coefficients);
    }    

    // extract point
    static pcl::PointCloud<PointType>::Ptr cloud_above_ground(new pcl::PointCloud<PointType>);
    static pcl::PointCloud<PointType>::Ptr cloud_ground(new pcl::PointCloud<PointType>);
    {    
        static pcl::ExtractIndices<PointType> extr;
        extr.setInputCloud(cloud);
        extr.setIndices(indices_ground);

        extr.setNegative(true);
        extr.filter(*cloud_above_ground);

        extr.setNegative(false);
        extr.filter(*cloud_ground);
    }

    // publish result
    static sensor_msgs::PointCloud2 msg_out;
    pcl::toROSMsg(*cloud_above_ground, msg_out);
    pub_pc_above_ground.publish(msg_out);
    pcl::toROSMsg(*cloud_ground, msg_out);
    pub_pc_ground.publish(msg_out);
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

    server.setCallback(dynamic_callback);
    auto sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 2, ros_callback);
    pub_pc_ground = nh.advertise<sensor_msgs::PointCloud2>("ground", 2);
    pub_pc_above_ground = nh.advertise<sensor_msgs::PointCloud2>("above_ground", 2);

    ros::spin();

    return 0;
}