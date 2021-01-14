**更详细的讲解参见 top 同学的 github 仓库**  
**此处需要 top 同学的 GitHub 链接**

# range_img
暂时没有注释和讲解

# bev_img
首先考虑如何将三维空间中的点投影到特定平面，这里想到了需要使用相机成像模型来实现，当然也可以粗暴的直接干掉 z 轴进行投影（这样做更快一些，但是变化比较少）  
确定好成像模型后对投影需要的参数进行推推到v，相关推导和图示都放在 pdf 中了，手癌晚期画的乱七八糟大家见谅😅  

确定好各种参数和相关推导就可以开始动手写代码了  
首先要过滤掉点云中数据不合理的点，然后确定点云的边界，这样好知道如何从物理坐标换算到成像平面的坐标，最后是对点云中的每一点进行求解，得到成像平面坐标后将对应位置的像素颜色设置为白色（0xffffff）就好

# eigenfeature
根据计算过程，首先求出目标点周围的 k 个最近点  
将这些点保存成一个 3xk 维度的矩阵，再计算这些点的协方差  
然后使用特征向量求解器得到协方差矩阵的三个特征值，将这三个特征值换算成 e  

> 到上面位置可以先编译一下，把求出来的东西打印出来看看正常不，如果没问题在开始后面的部分  

将求出来的 e 带入特征值的计算公式就能得到每个点的特征值了( •̀ ω •́ )y