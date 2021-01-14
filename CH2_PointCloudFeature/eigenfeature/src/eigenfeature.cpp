#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/filters/filter.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <fstream>

using Point = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<Point>;

void ros_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    PointCloud::Ptr pc(new PointCloud);
    std::vector<int> indices;
    pcl::KdTreeFLANN<Point> kdtree; // 用于搜索最近点


    // sensor_msgs::PointCloud2 -> pcl::PointCloud
    pcl::fromROSMsg(*msg, *pc);

    // is_dense 用于表示点云中的所有数据是否合法
    // 设置为 否，让后续的函数再检查一遍，把 Nan（not a number）
    // 这种不合法的数值全部去掉
    pc->is_dense = false;
    pcl::removeNaNFromPointCloud(*pc, *pc, indices);

    // 初始化 kd 树
    kdtree.setInputCloud(pc);

    // 提前将需要在循环中用到的变量初始化好，放置在循环中重复构造变量与析构，拖慢程序运行速度
    const int k = 20;                           // 临近点数量，根据作业要求设置为 20
    std::vector<int> point_idx(k);              // 用来保存临近点再原来点云中的下标
    std::vector<float> point_sq_dis(k);         // 用来保存临近点到目标点距离的平方
    std::vector<float> features(6);             // 用来保存六种点云特征
    std::vector<float> e(3);                    // 用来保存 k+1 个点经过 PCA 分析后得到的三个特征值计算得到的 e，从大到小排序
    std::ofstream file;                         // 输出计算结果的目标文件
    Eigen::Matrix<float, 3, 21> nearest_points; // 3x(k+1) 维的矩阵，用来保存点云中的点
    Eigen::Matrix3f covariance;                 // 用来保存协方差矩阵
    Eigen::Vector3f m, eigen_value;             // m 为 k+1 个点的质心，eigen_value 用来保存计算好的计算好的特征值

    // 打开文件，没有就凭空创建一个，如果有就删掉里面的内容，再写入新的
    // 一般不会出错
    file.open("wcm.txt");

    // pcl::PointCloud 中保存点的对象，我们用引用单独给他拿出来
    // 方便后续写代码
    auto& points = pc->points;
    for (size_t i = 0; i < pc->size(); i++)
    {
        // 每隔五个点计算一次特征值，作业没有要求这么做
        // 只是想这么做，希望能快点
        if(i%5 != 0) continue;

        // 重置 m，因为 m 需要累加，而其他的变量只需要赋值
        m = m.Zero();

        // 搜索目标点最近的几个点
        // https://pointclouds.org/documentation/classpcl_1_1_organized_neighbor_search.html#a3c18f38a4aad5fe6c05179906faf14cb
        kdtree.nearestKSearch(points[i], k, point_idx, point_sq_dis);

        // 累加搜索后的数据
        for (size_t j = 0; j < k; j++)
        {
            // 矩阵的块操作，将每个点作为列向量存入 nearest_points
            // http://eigen.tuxfamily.org/dox/group__TutorialBlockOperations.html
            nearest_points.col(j) << points[point_idx[j]].x, points[point_idx[j]].y, points[point_idx[j]].z;
            m[0] += points[point_idx[j]].x;
            m[1] += points[point_idx[j]].y;
            m[2] += points[point_idx[j]].z;
        }
        nearest_points.col(k) << points[i].x, points[i].y, points[i].z;
        m[0] += points[i].x;
        m[1] += points[i].y;
        m[2] += points[i].z;

        // 矩阵的广播操作，将每一列减去 k+1 个点的质心
        // http://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
        nearest_points.colwise() -= (m/(k+1));

        // 计算协方差矩阵
        covariance = nearest_points * nearest_points.transpose();

        // 对称矩阵求特征值
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
        eigen_value = solver.eigenvalues();

        // 矩阵的 reduction 操作，计算矩阵所有元素的和
        // http://eigen.tuxfamily.org/dox/group__TutorialReductionsVisitorsBroadcasting.html
        e[0] = eigen_value[2]/eigen_value.sum();
        e[1] = eigen_value[1]/eigen_value.sum();
        e[2] = eigen_value[0]/eigen_value.sum();

        // 计算点云特征
        features[0] = (e[0]-e[1])/e[0];
        features[1] = (e[1]-e[2])/e[0];
        features[2] = e[2]/e[0];
        features[3] = std::cbrt(std::accumulate(e.begin(), e.end(), 0))*3.0f;
        features[4] = -e[0] * std::log(e[0]) - e[1] * std::log(e[1]) - e[2] * std::log(e[2]);
        features[5] = 3.0f*e[2];

        // 将结果写入文件，空格分开，最有追加一个换行
        // 这种特殊的换行有清空缓冲区的效果
        for(auto& num : features)
            file << num << " ";
        file << std::endl;
    }
    // 关闭文件
    file.close();
    
    std::cout << "cal done" << std::endl;

    // 关闭节点
    // 下面是官方描述
    // Disconnects everything and unregisters from the master. 
    // It is generally not necessary to call this function, 
    // as the node will automatically shutdown when all NodeHandles destruct.
    // However, if you want to break out of a spin() loop explicitly, this function allows that.
    ros::shutdown();
    return;
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "aiimooc_wcm_node");
    ros::NodeHandle nh("~");

    auto sub_rslidar = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 2, ros_callback);

    ros::spin();
    return 0;
}
