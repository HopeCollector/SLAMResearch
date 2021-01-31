**详细讲解请移步 [top 同学的仓库](https://github.com/tangpan360/aiimooc_tangpan.git)**  

# lidarodom

这是一个基于 ndt 的激光里程计，使用的数据是由 [seg_rslidar](../CH4-Segmation/seg_rslidar/cluster) 发送的 `points_env`，这是一个去掉了环境中所有车辆行人的点云，只剩下了环境（体积特别大的的对象，以及不可识别的对象）

本身效率比较感人，可以考虑用 gpu 加速

# Registration

一个独立的项目，分别尝试了 icp， incremental-icp，ndt 三种注册方法，尝试着使用了现代 cmake 语法，大家有问题多提 issue 哈🤪

构建命令

```shell
$: cmake -S . -B build/debug -DCMAKE_BUILD_TYPE=Degbug
$: cmake --build build/debug
```