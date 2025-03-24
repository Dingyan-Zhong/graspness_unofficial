#pragma once
#include "cpu/vision.h"

#ifdef WITH_CUDA
#include "cuda/vision.h"
#include <c10/cuda/CUDAStream.h>
#include <ATen/ATen.h>
#endif

int knn(at::Tensor& ref, at::Tensor& query, at::Tensor& idx)
{
    // Check dimensions
    long batch = ref.size(0);
    long dim = ref.size(1);
    long k = idx.size(1);
    long ref_nb = ref.size(2);
    long query_nb = query.size(2);

    // Get raw pointers to tensor data
    float *ref_dev = ref.data_ptr<float>();
    float *query_dev = query.data_ptr<float>();
    int64_t *idx_dev = idx.data_ptr<int64_t>();  // Using int64_t instead of long for consistency

    if (ref.is_cuda()) {
#ifdef WITH_CUDA
        // Allocate temporary distance array on GPU
        at::Tensor dist_tensor = at::empty({ref_nb * query_nb}, 
                                         at::TensorOptions().dtype(at::kFloat).device(at::kCUDA));
        float* dist_dev = dist_tensor.data_ptr<float>();

        for (int b = 0; b < batch; b++) {
            knn_device(ref_dev + b * dim * ref_nb, ref_nb, 
                      query_dev + b * dim * query_nb, query_nb, 
                      dim, k, dist_dev, 
                      idx_dev + b * k * query_nb, 
                      c10::cuda::getCurrentCUDAStream());
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("error in knn: %s\n", cudaGetErrorString(err));
            AT_ERROR("aborting");
        }
        return 1;
#else
        AT_ERROR("Not compiled with GPU support");
#endif
    }

    // CPU path
    float *dist_dev = (float*)malloc(ref_nb * query_nb * sizeof(float));
    int64_t *ind_buf = (int64_t*)malloc(ref_nb * sizeof(int64_t));

    for (int b = 0; b < batch; b++) {
        knn_cpu(ref_dev + b * dim * ref_nb, ref_nb, 
                query_dev + b * dim * query_nb, query_nb, 
                dim, k, dist_dev, 
                idx_dev + b * k * query_nb, ind_buf);
    }

    free(dist_dev);
    free(ind_buf);

    return 1;
}
