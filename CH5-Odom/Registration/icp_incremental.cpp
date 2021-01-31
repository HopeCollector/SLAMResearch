#include <pcl/io/pcd_io.h>  //PCD相关头文件
#include <pcl/point_types.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/incremental_registration.h>

#include <boost/program_options.hpp>
#include <iostream>
#include <string>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace bpo = ::boost::program_options;
namespace fs = std::filesystem;
using PointT = pcl::PointXYZ;                 // x,y,z点
using PointCloudT = pcl::PointCloud<PointT>;  //点云　申明pcl::PointXYZ数据
using PointWithNormalT = pcl::PointNormal;

int main(int argc, char** argv) {
    // prase command line options
    std::string filepath_in;
    std::string filename_out;
    {
        bpo::options_description desc("Allowed options");
        desc.add_options()
            ("in,i", bpo::value<std::string>(&filepath_in), "input pcd file path !!! NOT FILE NAME !!!")
            ("out,o", bpo::value<std::string>(&filename_out), "output file name")
            ("help,h", "icp -i <input-filepath> -o <output-filename>");
        bpo::variables_map vm;

        try
        {
            bpo::store(bpo::parse_command_line(argc, argv, desc), vm);
        } catch (const std::exception& e) {
            std::cerr << e.what() << std::endl;
            std::cerr << desc << std::endl;
            return -1;
        }
        bpo::notify(vm);

        if(!(vm.count("in") && vm.count("out")))
        {
            std::cerr << desc << std::endl;
            return -1;
        }
    }

    // sort file name in right order
    std::vector<std::string> filenames_in;
    {
        if(!fs::exists(filepath_in))
        {
            std::cerr << "Cannot open dir " << filepath_in << std::endl;
            return -1;
        }
        for(const auto& entry : fs::directory_iterator(filepath_in))
        {
            filenames_in.push_back(entry.path());
        }
        std::sort(filenames_in.begin(), filenames_in.end());
    }

    // load file
    std::vector<PointCloudT::Ptr> clouds_in;
    {
        std::cout << "Loading pcd files" << std::endl;
        pcl::Indices indices_nan;
        for(const auto& name : filenames_in)
        {
            clouds_in.emplace_back(new PointCloudT);
            if(pcl::io::loadPCDFile(name, *clouds_in.back()) == -1)
            {
                std::cerr << "PCD File load error " << name << std::endl;
                return -1;
            }
            pcl::removeNaNFromPointCloud<PointT>(
                *clouds_in.back(), *clouds_in.back(), indices_nan);

            std::cout << name << " points num:" << clouds_in.back()->size() << std::endl;
        }

        std::cout << "\nSuccess load " << clouds_in.size() << " cloud" << std::endl;
    }

    // do icp
    PointCloudT::Ptr cloud_all(new PointCloudT);
    {
        // search method kdtree
        pcl::search::Search<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);

        // down sample
        pcl::VoxelGrid<PointT> vg;
        vg.setLeafSize(0.1, 0.1, 0.1);
        vg.setDownsampleAllData(true);

        // cal Normal
        pcl::NormalEstimation<PointT, PointWithNormalT> ne;
        ne.setKSearch(30);
        ne.setSearchMethod(tree);

        // icp
        pcl::IterativeClosestPointNonLinear<PointWithNormalT, PointWithNormalT>::Ptr icp_nl(new pcl::IterativeClosestPointNonLinear<PointWithNormalT, PointWithNormalT>);
        icp_nl->setTransformationEpsilon(1e-6);
        icp_nl->setMaxCorrespondenceDistance(0.05);
        icp_nl->setMaximumIterations(2);
        pcl::registration::IncrementalRegistration<PointWithNormalT> iicp;
        iicp.setRegistration(icp_nl);

        pcl::PointCloud<PointWithNormalT>::Ptr cloud_normal(new pcl::PointCloud<PointWithNormalT>);
        for (auto& cloud_in : clouds_in) {
            vg.setInputCloud(cloud_in);
            vg.filter(*cloud_in);

            ne.setInputCloud(cloud_in);
            ne.compute(*cloud_normal);
            pcl::copyPointCloud(*cloud_in, *cloud_normal);

            iicp.registerCloud(cloud_normal);
            pcl::transformPointCloud(*cloud_in, *cloud_in, iicp.getAbsoluteTransform());
            *cloud_all += *cloud_in;
        }

        // down samle cloud_all, because too many duplicate points
        vg.setInputCloud(cloud_all);
        vg.filter(*cloud_all);
    }

    pcl::io::savePCDFile(filename_out, *cloud_all);

    return 0;
}
