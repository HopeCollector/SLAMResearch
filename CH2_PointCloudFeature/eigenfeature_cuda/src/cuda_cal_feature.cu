#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda.h>

// cuda 需要再其运算单元中放入进行运算的函数，我们用
// __global__ 进行指定，这个函数被称为 kernel
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#programming-model
__global__
void _cal_feature(float* e, float* features)
{
    switch (threadIdx.x)
    {
    case 0:
        features[0] = (e[0] - e[1]) / e[0];
        break;

    case 1:
        features[1] = (e[1] - e[2]) / e[0];
        break;

    case 2:
        features[2] = e[2] / e[0];
        break;

    case 3:
        features[3] = std::cbrt(e[0]*3.0f + e[1]*3.0f + e[2]*3.0f);
        break;

    case 4:
        features[4] = -e[0] * std::log(e[0]) - e[1] * std::log(e[1]) - e[2] * std::log(e[2]);
        break;

    case 5:
        features[5] = 3.0f * e[2];
        break;
    
    default:
        break;
    }
}

void cal_feature(float*e, float*features)
{
    // cuda 自己新定义的运算符 <<<blockNum，threadNumEachBlock, memSize>>>
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#execution-configuration
    _cal_feature<<<1,6>>>(e,features);
}