**详细讲解请移步 [top 同学的仓库](https://github.com/tangpan360/aiimooc_tangpan.git)**  

# transform

首先进入 `transform` 文件夹，然后按照下面的构建步骤构建可执行文件  

```shell
mkdir build && cd build
cmake ..
make
```  

构建成功后使用下面的指令执行代码即可在命令得到输出，同时得到一个 transform_xxx.txt 文件, 记得修改成自己的名字！！！！  

```shell
./transform
```

# kitti_ros / kitti_lidar_camera  

修改过的代码，可以运行在 `Ubuntu 20.04` 上，需要修改 `demo.launch` 下的 `kitti_data_path` 变量，当前版本的运行效率非常感人，等我研究出更高效的方法就发上来 🤦‍

现在代码还没有注释，后续会慢慢补上，可能的话还会出一个 cpp 版本