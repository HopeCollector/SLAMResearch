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
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include <iostream>
#include <vector>
#include <limits>
#include "seg_rslidar/SegRslidarConfig.h"

using PointT = pcl::PointXYZI;
using PointCloudT = pcl::PointCloud<PointT>;
using pcl::Normal;

float smothness_threshold;
float cluster_tolerance;
ros::Publisher pub_pc_env;
ros::Publisher pub_pc_obj;
ros::Publisher pub_marker;

void ros_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    // load pointcloud
    static PointCloudT::Ptr cloud(new PointCloudT);
    static pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    pcl::fromROSMsg(*msg, *cloud);

    // downsample
    {
        static pcl::VoxelGrid<PointT> vg;
        static bool is_init = false;
        if(!is_init)
        {
            vg.setDownsampleAllData(true);
            vg.setLeafSize(0.5, 0.5, 0.5);
            vg.setInputCloud(cloud);
            is_init = true;
        }
        vg.filter(*cloud);
    }

    // segment
    static std::vector<pcl::PointIndices> clusters;
    {
        static pcl::ConditionalEuclideanClustering<PointT> cgc;
        static bool is_init = false;
        if(!is_init)
        {
            cgc.setSearchMethod(tree);
            cgc.setMinClusterSize(cloud->size() / 1000);
            cgc.setMaxClusterSize(cloud->size()/5);
            cgc.setInputCloud(cloud);
            cgc.setConditionFunction([](PointT a, PointT b, float sq_dis) { return (b.intensity < 254.0); });
            is_init = true;
        }

        clusters.clear();
        cgc.setClusterTolerance(cluster_tolerance);
        cgc.segment(clusters);
    }

    // extract bounding box & env/obj points
    static std::vector<std::pair<Eigen::Vector4f, Eigen::Vector4f>> bboxes;
    static PointCloudT::Ptr cloud_env(new PointCloudT);
    static PointCloudT::Ptr cloud_obj(new PointCloudT);
    {
        static std::vector<bool> is_env_point;

        // get enough space for data
        bboxes.clear();
        cloud_env->clear();
        if (cloud_env->points.capacity() < cloud->size()) cloud_env->reserve(cloud->size());
        cloud_obj->clear();
        if (cloud_obj->points.capacity() < cloud->size()) cloud_obj->reserve(cloud->size());
        is_env_point.resize(cloud->size());
        std::fill(is_env_point.begin(), is_env_point.end(), true);

        Eigen::Vector4f p_min, p_max;
        for (const auto& cluster : clusters) {
            pcl::getMinMax3D(*cloud, cluster, p_min, p_max);
            auto v = (p_max-p_min).head(3);
            if (v.dot(v) < 125 )
            {
                bboxes.push_back({p_min, p_max});
                for(auto idx : cluster.indices)
                    is_env_point[idx] = false;
            }
        }

        PointT p;
        for (int i = 0; i < cloud->size(); i++)
        {
            memcpy(reinterpret_cast<char*>(p.data), reinterpret_cast<char*>(cloud->at(i).data), 3 * sizeof(float));
            if (is_env_point[i])
            {
                cloud_env->emplace_back(p);
            }else
            {
                cloud_obj->emplace_back(p);
            }
        }
    }

    // gen markers
    static visualization_msgs::MarkerArray mks;
    {
        if(mks.markers.size() > bboxes.size())
        {
            for (size_t i = bboxes.size(); i < mks.markers.size(); i++)
            {
                mks.markers[i].action = visualization_msgs::Marker::DELETE;
            }
        }else
        {
            mks.markers.resize(bboxes.size());
        }

        for (int i = 0; i < bboxes.size(); i++)
        {
            auto& [p_min, p_max] = bboxes[i];
            auto& mk = mks.markers[i];
            mk.header = msg->header;
            mk.id = i;
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
            mk.pose.position.x = (p_min[0]+p_max[0])/2.0;
            mk.pose.position.y = (p_min[1]+p_max[1])/2.0;
            mk.pose.position.z = (p_min[2]+p_max[2])/2.0;
            mk.scale.x = (p_max[0]-p_min[0]);
            mk.scale.y = (p_max[1]-p_min[1]);
            mk.scale.z = (p_max[2]-p_min[2]);
        }
    }

    // publish results
    {
        static sensor_msgs::PointCloud2 msg_out_pc_env;
        static sensor_msgs::PointCloud2 msg_out_pc_obj;
        static bool is_init = false;
        if(!is_init)
        {
            cloud_env->header = cloud->header;
            cloud_obj->header = cloud->header;
            is_init = true;
        }

        pcl::toROSMsg(*cloud_env, msg_out_pc_env);
        pub_pc_env.publish(msg_out_pc_env);

        pcl::toROSMsg(*cloud_obj, msg_out_pc_obj);
        pub_pc_obj.publish(msg_out_pc_obj);

        pub_marker.publish(mks);
    }

}

void dynamic_callback(seg_rslidar_param::SegRslidarConfig& cfg, uint32_t level)
{
    cluster_tolerance = cfg.cluster_tolerance;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "cluster");
    ros::NodeHandle nh("~");
    dynamic_reconfigure::Server<seg_rslidar_param::SegRslidarConfig> server;

    nh.param("cluster_tolerance", cluster_tolerance, 1.0f);

    server.setCallback(dynamic_callback);
    auto sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("/ground_removal/points_all", 1, ros_callback);
    pub_pc_env = nh.advertise<sensor_msgs::PointCloud2>("/cluster/points_env", 1);
    pub_pc_obj = nh.advertise<sensor_msgs::PointCloud2>("/cluster/points_obj", 1);
    pub_marker = nh.advertise<visualization_msgs::MarkerArray>("/cluster/marker", 1);

    ros::spin();

    return 0;
}