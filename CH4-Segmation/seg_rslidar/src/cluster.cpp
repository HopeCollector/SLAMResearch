#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/PointIndices.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>

#include <iostream>
#include <vector>
#include "seg_rslidar/SegRslidarConfig.h"

using PointType = pcl::PointXYZ;
using pcl::Normal;

float smothness_threshold = 10.0/180.0*M_PI;
float curvature_threshold = 1.0;
ros::Publisher pub_pc;
ros::Publisher pub_pc_ground;
ros::Publisher pub_marker;
pcl::PointCloud<PointType>::Ptr cloud_ground(new pcl::PointCloud<PointType>);

void ros_callback_ground(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    pcl::fromROSMsg(*msg, *cloud_ground);
}

void ros_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    // load pointcloud
    static pcl::PointCloud<PointType>::Ptr cloud(new pcl::PointCloud<PointType>);
    pcl::fromROSMsg(*msg, *cloud);

    // cal normal
    static pcl::PointCloud<Normal>::Ptr normals(new pcl::PointCloud<Normal>);
    static pcl::search::Search<PointType>::Ptr tree(new pcl::search::KdTree<PointType>);
    {
        static pcl::NormalEstimation<PointType, Normal> ne;
        ne.setInputCloud(cloud);
        ne.setSearchMethod(tree);
        ne.setKSearch(30);
        ne.compute(*normals);
    }

    // segment
    static std::vector<pcl::PointIndices> clusters;
    static pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_color;
    {
        static pcl::RegionGrowing<PointType, Normal> reg;
        clusters.clear();
        reg.setMinClusterSize(50);
        reg.setMaxClusterSize(10000);
        reg.setSearchMethod(tree);
        reg.setNumberOfNeighbours(50);
        reg.setInputCloud(cloud);
        reg.setInputNormals(normals);
        reg.setSmoothnessThreshold(smothness_threshold);
        reg.setCurvatureThreshold(curvature_threshold);
        reg.extract(clusters);
        cloud_color = reg.getColoredCloud();
    }

    // publish
    {
        static visualization_msgs::MarkerArray mks;
        static visualization_msgs::Marker mk;
        static Eigen::Vector4f p_min, p_max;

        mk.header = msg->header;
        mk.ns = "marker";
        mk.type = visualization_msgs::Marker::CUBE;
        mk.action = visualization_msgs::Marker::MODIFY;
        mk.pose.orientation.x = 0;
        mk.pose.orientation.y = 0;
        mk.pose.orientation.z = 0;
        mk.pose.orientation.w = 1;
        mk.color.r = 1.0;
        mk.color.g = 0;
        mk.color.b = 0;
        mk.color.a = 0.15;

        mks.markers.clear();
        mks.markers.reserve(clusters.size());
        for(int i = 0, cnt = 0; i < clusters.size(); i++)
        {
            pcl::getMinMax3D(*cloud, clusters[i], p_min, p_max);
            auto tmp = Eigen::Vector3f((p_max-p_min).head<3>());
            if(tmp.dot(tmp) > 225) continue;
            
            mk.id = cnt; cnt++;
            mk.pose.position.x = (p_min[0]+p_max[0])/2.0;
            mk.pose.position.y = (p_min[1]+p_max[1])/2.0;
            mk.pose.position.z = (p_min[2]+p_max[2])/2.0;
            mk.scale.x = (p_max[0]-p_min[0]);
            mk.scale.y = (p_max[1]-p_min[1]);
            mk.scale.z = (p_max[2]-p_min[2]);
            mks.markers.emplace_back(mk);
        }

        static sensor_msgs::PointCloud2 msg_out_pc, msg_out_pc_ground;
        pcl::toROSMsg(*cloud_color, msg_out_pc);
        msg_out_pc.header = msg->header;
        pub_pc.publish(msg_out_pc);

        pub_marker.publish(mks);

        if(cloud_ground->size())
        {
            pcl::toROSMsg(*cloud_ground, msg_out_pc_ground);
            pub_pc_ground.publish(msg_out_pc_ground);
        }
    }

}

void dynamic_callback(seg_rslidar_param::SegRslidarConfig& cfg, uint32_t level)
{
    smothness_threshold = cfg.smothness_threshold;
    curvature_threshold = cfg.curvature_threshold;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cluster");
    ros::NodeHandle nh("~");
    dynamic_reconfigure::Server<seg_rslidar_param::SegRslidarConfig> server;

    server.setCallback(dynamic_callback);
    auto sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("above_ground", 2, ros_callback);
    auto sub_pc_ground = nh.subscribe<sensor_msgs::PointCloud2>("ground", 2, ros_callback_ground);
    pub_pc = nh.advertise<sensor_msgs::PointCloud2>("classified_points", 2);
    pub_pc_ground = nh.advertise<sensor_msgs::PointCloud2>("slow_ground", 2);
    pub_marker = nh.advertise<visualization_msgs::MarkerArray>("marker", 2);

    ros::spin();

    return 0;
}