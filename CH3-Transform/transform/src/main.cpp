#include <iostream>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <string>
#include <fstream>

using Eigen::AngleAxisf;
using Eigen::Matrix3f;
using Eigen::Vector3f;
using Eigen::Quaternionf;
using Eigen::Transform;

int main()
{
    const float angle_x = 30.0f * M_PI / 180.0f;
    const float angle_y = 20.0f * M_PI / 180.0f;
    const float angle_z = 10.0f * M_PI / 180.0f;
    const Vector3f target(1, 2, 3);
    const Vector3f bias(3, 2, 1);
    Matrix3f fix_axis_m;
    Matrix3f unfix_axis_m;
    std::ofstream file;

    file.open("transform_wcm.txt");

    // 计算固定坐标系下的旋转矩阵
    {
        // Eigen 自带的轴角表示系统，传入坐标轴和旋转角度即可
        // 由于 AngleAxisf 重载了乘法运算，所以合并多个旋转矩阵只要用乘法就行
        // http://eigen.tuxfamily.org/dox/classEigen_1_1AngleAxis.html
        Matrix3f m = AngleAxisf(angle_z, Vector3f::UnitZ())
            * AngleAxisf(angle_y, Vector3f::UnitY())
            * AngleAxisf(angle_x, Vector3f::UnitX()).toRotationMatrix();

        std::cout << "\n固定坐标系下的旋转矩阵：" << std::endl
            << m << std::endl;
        fix_axis_m = m;

        AngleAxisf res;
        // 从旋转矩阵中得到轴角表示
        // http://eigen.tuxfamily.org/dox/classEigen_1_1AngleAxis.html#a2e35689645f69ba886df1a0a14b76ffe
        res.fromRotationMatrix(fix_axis_m);
        file << "固定坐标系：" << std::endl
             << "旋转轴：(" << res.axis().transpose() << "), 旋转角" << res.angle() * 180.0f / M_PI << std::endl
             << "四元数表示：[" << Quaternionf(res).vec().transpose() << "]" << std::endl;
    }

    // 计算不固定坐标系下的旋转矩阵
    {
        Vector3f axis_x = Vector3f::UnitX();
        Vector3f axis_y = Vector3f::UnitY();
        Vector3f axis_z = Vector3f::UnitZ();
        Matrix3f m = Matrix3f::Identity();

        // 这里的 lambda 函数对三个坐标轴都进行了旋转
        // 注意函数的捕获列表，& 表示捕获作用域内所有变量的引用
        // 所以在函数内我们可以直接修改函数外变量的值
        auto rotate_axises = [&](const AngleAxisf rotate) {
            axis_x = rotate * axis_x;
            axis_y = rotate * axis_y;
            axis_z = rotate * axis_z;
            m = rotate * m;
        };

        // 从 x 开始旋转
        rotate_axises(AngleAxisf(angle_x, axis_x));
        rotate_axises(AngleAxisf(angle_y, axis_y));
        rotate_axises(AngleAxisf(angle_z, axis_z));

        std::cout << "\n不固定坐标系下的旋转矩阵：" << std::endl
                  << m << std::endl;
        unfix_axis_m = m;

        AngleAxisf res;
        res.fromRotationMatrix(unfix_axis_m);
        file << "\n不固定坐标系：" << std::endl
             << "旋转轴：(" << res.axis().transpose() << "), 旋转角" << res.angle() * 180.0f / M_PI << std::endl
             << "四元数表示：[" << Quaternionf(res).vec().transpose() << "]" << std::endl;
    }

    // 计算向量的齐次变换
    {
        // 使用 Transform 表示齐次变换
        // http://eigen.tuxfamily.org/dox/group__TutorialGeometry.html#title2
        Transform<float, 3, Eigen::TransformTraits::Affine> t;
        t = AngleAxisf(angle_z, Vector3f::UnitZ());
        // 等价于 t = t * AngleAxisf(angle_y, Vector3f::UnitY())
        t.rotate(AngleAxisf(angle_y, Vector3f::UnitY())); 
        t.rotate(AngleAxisf(angle_x, Vector3f::UnitX()));
        // 等价于 t = t*() + bias
        // 括号表示要旋转的向量
        t.pretranslate(bias);
        std::cout << std::endl << t.matrix() << std::endl;

        std::cout << "\n固定坐标下，齐次变换目标向量结果为：" << std::endl
            << t*target << std::endl;

        file << "\n固定坐标下，齐次变换目标向量结果为：("
             << (t * target).transpose() << ")" << std::endl;

        // 从旋转矩阵构造四元数
        // http://eigen.tuxfamily.org/dox/classEigen_1_1QuaternionBase.html#title30
        Quaternionf q(unfix_axis_m);
        std::cout << "\n不固定坐标系下，齐次变换目标向量结果为：" << std::endl
            << (q.toRotationMatrix()*target) + bias << std::endl;

        file << "\n不固定坐标下，齐次变换目标向量结果为：("
             << ((q.toRotationMatrix() * target) + bias).transpose() << ")" << std::endl;
    }
    file.close();

    // 手搓一个固定坐标系下的旋转矩阵，证明 Eigen 库的正确性
    {
        using std::sin;
        using std::cos;
        Matrix3f m = Matrix3f::Identity();
        Matrix3f tmp;

        // 绕 x 轴旋转
        tmp << 1, 0, 0,
            0, cos(angle_x), -sin(angle_x),
            0, sin(angle_x), cos(angle_x);
        m = tmp * m;

        // 绕 y 轴旋转
        tmp << cos(angle_y), 0, sin(angle_y),
            0, 1, 0,
            -sin(angle_y), 0, cos(angle_y);
        m = tmp * m;

        // 绕 z 轴旋转
        tmp << cos(angle_z), -sin(angle_z), 0,
            sin(angle_z), cos(angle_z), 0,
            0, 0, 1;
        m = tmp*m;
        std::cout << std::endl << m*target + bias << std::endl;
    }

    return 0;
}