**详细讲解请移步 [top 同学的仓库](https://github.com/tangpan360/aiimooc_tangpan.git)**  

# 分割点云  
代码中使用的是区域增长分割，可以替换成其他喜欢的分割，下面是数据的处理流程

- 原始数据广播
- 地面识别（ground_removal)
  - 地面点云
  - 地面以外的点云
- 对地面以外的点云进行区域增长分割 (cluster)

下载后记得给 cfg 文件添加可执行权限