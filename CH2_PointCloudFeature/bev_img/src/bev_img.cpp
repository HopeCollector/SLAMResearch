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
    // 将常用的变量设置为静态变量，这样每次进入回调函数都不需要重新初始化这些变量
    // 也不需要重新为它们分配内存空间，可以稍微提升运行效率
    static PointCloud pc;
    static sensor_msgs::Image out_img;
    static std::vector<int> idies; // 用来保存数据为 Nan 的点在点云中的下标，某些时候肯能有用
    pcl::fromROSMsg(*msg, pc);

    // is_dense 用于确定点云中的点是否全部有效（没有数据是 Nan）
    // 设置为 false 可以用自带函数再检查一下，避免运算时出错不好追查
    pc.is_dense = false;
    pcl::removeNaNFromPointCloud(pc, pc, idies);

    // 获取点云的边界，bound 用来控制边界的最大范围
    // https://pointclouds.org/documentation/group__common.html#ga3166f09aafd659f69dc75e63f5e10f81
    static Point min, max, bound;
    pcl::getMinMax3D(pc, min, max);
    bound.x = std::max(std::abs(min.x), std::abs(max.x));
    bound.y = std::max(std::abs(min.y), std::abs(max.y));
    bound.z = std::max(std::abs(min.z), std::abs(max.z));

    // project cloud to camera plane
    {
        // 将这些不会变的变量设置为静态常量可以省去重复初始化的步骤
        // const 可以保证在不小心修改变量时编译器能轻松找到错误在哪
        // 不使用宏定义是因为这样可以强化变量类型，让编译器帮忙做类型检查
        // 而且看着好看！（个人觉得这个最重要( •̀ ω •́ )y
        static const float img_width = 1920.0f;         // 图像宽度，这个其实随便怎么设定都可以
        static const float img_height = 1080.0f;        // 图像高度
        static const float camera_height = 100.0f;      // 相机高度，在 pdf 中有讲
        static const float focal = 1.0f;                // 相机到成像平面的距离（焦距）
        static const float cv = img_height / 2.0f;      // 成像平面的中心点 v
        static const float cu = img_width / 2.0f;       // 成像平面的中心点 u
        static float x2v, y2u;                          // 将成像平面上的点从 xy 坐标系转换到 uv 坐标系的放缩系数

        // 将点云的边界裁剪成 16:9 的大小，保证所有点都能投射到成像平面上
        if((bound.y / bound.x) < img_width/img_height)
            bound.y = bound.x * img_width/img_height;
        else
            bound.x = bound.y * img_height/img_width;

        // 计算放缩系数，具体推到在 pdf
        x2v = -(cv*(camera_height-bound.z) / bound.x / focal);
        y2u = -(cu*(camera_height-bound.z) / bound.y / focal);

        // lambda 函数 [捕获列表](参数列表)->返回值{具体函数语句}
        // 其中返回值一般可省略，编译器能从函数语句中自动推导出来
        // 这里的函数用来将一个三维的坐标换算到成像平面的 uv 坐标
        // 具体推到详见 pdf
        auto getPixel = [&](Point& p){
            Pixel pixel;
            pixel.v = static_cast<int>(p.x * focal / (camera_height - p.z) * x2v + cv);
            pixel.u = static_cast<int>(p.y * focal / (camera_height - p.z) * y2u + cu);
            return pixel;
        };

        // 这里参考了 pcl_conversion::toROSMsg 的写法
        // http://docs.ros.org/en/indigo/api/pcl_conversions/html/pcl__conversions_8h_source.html#l00506
        static const uint32_t color_white = 0x00ffffff;
        out_img.encoding = "bgr8";

        // 之所以 +1，是因为有的点确实会映射到图像边界的位置，而 cpp 的数组下标是 0~k-1，所以多申请一个变成 0~k
        out_img.width = img_width+1; 
        out_img.height = img_height+1;

        // step 表示一行数据有多长（单位：字节）
        // 每个数据由 rgb 三个数据组成，而每个颜色由一个 uint8_t 组成
        // uint8_t 表示无符号整型，一个数据需要 8 位
        // 具体关于各种数据类型的定义在下面链接，建议写代码的时候多用这种定义明确的写法
        // 而不是只写一个 int ，因为在不同的机器上对这样一种类型要用多少位，规定是不一样的
        // https://en.cppreference.com/w/cpp/types/integer
        out_img.step = out_img.width * sizeof(uint8_t) * 3;
        out_img.data.resize(out_img.step * out_img.height);
        std::fill_n(out_img.data.begin(), out_img.data.size(), 0);
        for(auto& point : pc.points)
        {
            auto pos = getPixel(point);

            // 进行一些高危操作时，可以将这些操作放到 try-catch 语句中中用于捕捉错误
            // 当操作出问题时方便定位
            try
            {
                uint8_t *pixel = &(out_img.data[pos.v * out_img.step + pos.u * 3]);

                // 将数据从 color_white 开始的三个字节，拷贝到 pixel 指向的位置
                // https://en.cppreference.com/w/cpp/string/byte/memcpy
                memcpy(pixel, &color_white, 3 * sizeof(uint8_t));
            }
            catch(...)
            {
                std::cerr << point << " -> "
                            << "(" << pos.u << ", " << pos.v << ")\n";
                assert(0);
            }
            
        }
    }

    // publish img msg
    pub_img.publish(out_img);
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "bev_img_node");
    ros::NodeHandle nh("~");

    auto sub_pc = nh.subscribe<sensor_msgs::PointCloud2>("/rslidar_points", 2, ros_callback);
    pub_img = nh.advertise<sensor_msgs::Image>("/img_bev", 1);

    ros::spin();

    return 0;
}
