#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/ndt.h>
#include <pcl/registration/incremental_registration.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>

using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

ros::Publisher pub_cloud_map;

void ros_callback_regist(const sensor_msgs::PointCloud2::ConstPtr& msg)
{
    static PointCloudT::Ptr cloud_env(new PointCloudT);
    static PointCloudT::Ptr cloud_map(new PointCloudT);
    static sensor_msgs::PointCloud2::Ptr msg_out_pc(new sensor_msgs::PointCloud2);
 
    // load point cloud
    // static pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_in(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud_env);

    // icp
    {
        pcl::NormalDistributionsTransform<PointT, PointT>::Ptr ndt(new pcl::NormalDistributionsTransform<PointT, PointT>);
        static pcl::registration::IncrementalRegistration<PointT> iicp;
        static bool is_init = false;
        if (!is_init)
        {
            ndt->setTransformationEpsilon(0.03);
            ndt->setStepSize(0.3);
            ndt->setResolution(3);
            ndt->setMaximumIterations(30);
            iicp.setRegistration(ndt);
            is_init = true;
        }
        if(!iicp.registerCloud(cloud_env))
        {
            ROS_WARN("Cannot register cloud!");
            return;
        }
        pcl::transformPointCloud(*cloud_env, *cloud_env, iicp.getAbsoluteTransform());
        *cloud_map += *cloud_env;
    }
    

    // filter
    {
        static pcl::VoxelGrid<PointT> vg;
        static bool is_init = false;
        if(!is_init)
        {
            vg.setLeafSize(1.0, 1.0, 1.0);
            vg.setInputCloud(cloud_map);
            is_init = true;
        }
        vg.filter(*cloud_map);
    }

    cloud_map->header = cloud_env->header;
    pcl::toROSMsg(*cloud_map, *msg_out_pc);
    pub_cloud_map.publish(msg_out_pc);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "lidarodom");
    ros::NodeHandle nh("~");

    auto sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("/cluster/points_env", 1, ros_callback_regist);
    pub_cloud_map = nh.advertise<sensor_msgs::PointCloud2>("/lidarodom/points_map", 1);

    ros::spin();
    return 0;
}