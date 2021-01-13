#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/common.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>

using Point = pcl::PointXYZI;
using PointCloud = pcl::PointCloud<Point>;
using PCLImag = pcl::PCLImage;
struct Pixel {
    int u, v;

    Pixel(){}
    Pixel(int _u, int _v) : v(_v), u(_u) {}
};

ros::Publisher pub_img;

void ros_callback(const sensor_msgs::PointCloud2ConstPtr& msg)
{
    static PointCloud pc;
    static sensor_msgs::Image out_img;
    static pcl::Indices idies;
    pcl::fromROSMsg(*msg, pc);
    pc.is_dense = false;
    pcl::removeNaNFromPointCloud(pc, pc, idies);

    // get bounding box of cloud
    static Point min, max, bound;
    pcl::getMinMax3D(pc, min, max);
    bound.x = std::max(std::abs(min.x), std::abs(max.x));
    bound.y = std::max(std::abs(min.y), std::abs(max.y));
    bound.z = std::max(std::abs(min.z), std::abs(max.z));

    // project cloud to camera plane
    {
        static const float img_width = 1920.0f;
        static const float img_height = 1080.0f;
        static const float camera_height = 100.0f;
        static const float focal = 1.0f;
        static const float cv = img_height / 2.0f;
        static const float cu = img_width / 2.0f;
        static float x2v, y2u;

        if((bound.y / bound.x) < img_width/img_height)
            bound.y = bound.x * img_width/img_height;
        else
            bound.x = bound.y * img_height/img_width;

        x2v = -(cv*(camera_height-bound.z) / bound.x / focal);
        y2u = -(cu*(camera_height-bound.z) / bound.y / focal);

        auto getPixel = [&](Point& p){
            Pixel pixel;
            pixel.v = static_cast<int>(p.x * focal / (camera_height - p.z) * x2v + cv);
            pixel.u = static_cast<int>(p.y * focal / (camera_height - p.z) * y2u + cu);
            return pixel;
        };

        static const uint32_t color_white = 0x00ffffff;
        out_img.encoding = "bgr8";
        out_img.width = img_width+1;
        out_img.height = img_height+1;
        out_img.step = out_img.width * sizeof(uint8_t) * 3;
        out_img.data.resize(out_img.step * out_img.height);
        std::fill_n(out_img.data.begin(), out_img.data.size(), 0);
        for(auto& point : pc.points)
        {
            auto pos = getPixel(point);
            try
            {
                uint8_t *pixel = &(out_img.data[pos.v * out_img.step + pos.u * 3]);
                memcpy(pixel, &color_white, 3 * sizeof(uint8_t));
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                std::cerr << point << " -> "
                            << "(" << pos.u << ", " << pos.v << ")\n";
            }
            
        }
    }

    // publish img msg
    pub_img.publish(out_img);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "gen_bev_img_node");
    ros::NodeHandle nh("~");

    auto sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 2, ros_callback);
    pub_img = nh.advertise<sensor_msgs::Image>("/img_bev", 1);

    ros::spin();

    return 0;
}