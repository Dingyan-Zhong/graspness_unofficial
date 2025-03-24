#pragma once
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

void knn_device(float* ref_dev, int ref_width,
    float* query_dev, int query_width,
    int height, int k, float* dist_dev, 
    int64_t* ind_dev,  // Changed from long* to int64_t*
    cudaStream_t stream);