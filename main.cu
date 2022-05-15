#include <iostream>
#include <unistd.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define HALF
// #define FLOAT

#if defined(FLOAT)
    #define converter(f) f
    #define converter_back(f) f
#elif defined(HALF)
    // #define TENSOR_CORE
    #define converter(f) __float2half(f)
    #define converter_back(f) __half2float(f)
#endif

using namespace std;

#define n 256
#define m 256

float max_val = float((n - 1)*m + m-1);

template <typename T>
void PutData(T *host_a)
{
    T val;
    float f_val;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            f_val = float(i*m + j);
            f_val /= max_val;
            val = converter(f_val);
            host_a[i*m + j] = val;
        }   
    }
}

template <typename T>
void PrintMatrix(T *a, const int rows, const int columns)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < columns; j++)
            printf("%f ", converter_back(a[i*columns + j]));
        printf("\n");   
    }
    printf("\n"); 
}

template <typename T>
void PrintDeviceMatrix(T *dev, const int rows, const int columns)
{
    T *host;
    host = (T*)malloc(rows * columns * sizeof(T));
    cudaMemcpy(host, dev, rows * columns * sizeof(T), cudaMemcpyDeviceToHost);
    PrintMatrix(host, rows, columns);
    free(host);
}

template <typename T>
__global__ void SetValTo1(T *dev, const int rows, const int columns)
{
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    float unit = 1.0;
    if(id < rows * columns)
        dev[id] = converter(unit);
}

#if defined(HALF)
    int main(int argc, char* argv[])
    {
        printf("HALF\n");
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        const int BlockCount = ceil((double(m * n)) / 1024.0);
        half *host_a = 0;

        half *dev_a;
        half *dev_e1;
        half *dev_c1;
        half *dev_e2;
        half *dev_res;

        cublasHandle_t handle;
        cublasStatus_t cublasStat = cublasCreate(&handle);
        const half alpha = 1.0f;
        const half beta = 0.0f;
        half reduce_sum = 1.0f;

        // for (int i = 51008; i < 51008 + 2 * 1e+4; ++i)
        //     printf("%f\n", __half2float(__int2half_rn(i)));

        host_a = (half*)malloc(n * m * sizeof(half));

        cudaMalloc((void**)&dev_a, n * m * sizeof(half));
        cudaMalloc((void**)&dev_c1, n * n * sizeof(half));
        cudaMalloc((void**)&dev_e1, m * n * sizeof(half));
        cudaMalloc((void**)&dev_e2, n * m * sizeof(half));
        cudaMalloc((void**)&dev_res, m * n * sizeof(half));

        PutData(host_a);

        cudaMemcpy(dev_a, host_a, n * m * sizeof(half), cudaMemcpyHostToDevice);

        SetValTo1<<<BlockCount, 1024 >>>(dev_e1, m, n);
        SetValTo1<<<BlockCount, 1024 >>>(dev_e2, m, n);

        // PrintDeviceMatrix(dev_a, n, m);

        cudaStreamSynchronize(0);

        // PrintDeviceMatrix(dev_c1, n, n);
        // PrintDeviceMatrix(dev_e1, m, n);

        #ifdef TENSOR_CORE
            cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        #else
            cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH);
        #endif

        cudaEventRecord(start);
        #ifdef TENSOR_CORE
            printf("TENSOR_CORE\n");
            cublasStat = cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            n, n, m,
            &alpha,
            dev_e1, CUDA_R_16F, m,
            dev_a, CUDA_R_16F, n,
            &beta,
            dev_c1, CUDA_R_16F, n,
            CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);


            cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
            m, n, n,
            &alpha,
            dev_e2, CUDA_R_16F, m,
            dev_c1, CUDA_R_16F, n,
            &beta,
            dev_res, CUDA_R_16F, m,
            CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        #else
            printf("DEFAULT_MATH\n");
            cublasStat = cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
            n, n, m,
            &alpha,
            dev_e1, m,
            dev_a, n,
            &beta,
            dev_c1, n);

            cublasStat = cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
            m, n, n,
            &alpha,
            dev_e2, m,
            dev_c1, n,
            &beta,
            dev_res, m);
        #endif
        cudaEventRecord(stop);

        cudaMemcpy(&reduce_sum, dev_res, sizeof(half), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);

        float GPUtime = 0;
        cudaEventElapsedTime(&GPUtime, start, stop);
        GPUtime *= 0.001;

        printf("Reduce elapsed time >>> %e sec\n", GPUtime);
        printf("Reduce sum is = %f\n", converter_back(reduce_sum) * max_val);


        cudaFree(dev_a);
        cudaFree(dev_e1);
        cudaFree(dev_c1);
        cudaFree(dev_e2);
        cudaFree(dev_res);

        free(host_a);

        return 0;
    }

#elif defined(FLOAT)
    int main(int argc, char* argv[])
    {
        printf("Float\n");

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        const int BlockCount = ceil((double(m * n)) / 1024.0);

        float *host_a = 0;
        float *dev_a;
        float *dev_e1;
        float *dev_c1;
        float *dev_e2;
        float *dev_res;

        const float alpha = 1.0f;
        const float beta = 0.0f;
        float reduce_sum = 1.0f;

        cublasHandle_t handle;
        cublasStatus_t cublasStat = cublasCreate(&handle);

        host_a = (float*)malloc(n * m * sizeof(float));

        cudaMalloc((void**)&dev_a, n * m * sizeof(float));
        cudaMalloc((void**)&dev_c1, n * n * sizeof(float));
        cudaMalloc((void**)&dev_e1, m * n * sizeof(float));
        cudaMalloc((void**)&dev_e2, n * m * sizeof(float));
        cudaMalloc((void**)&dev_res, m * n * sizeof(float));

        PutData(host_a);

        cudaMemcpy(dev_a, host_a, n * m * sizeof(float), cudaMemcpyHostToDevice);

        SetValTo1<<<BlockCount, 1024 >>>(dev_e1, m, n);
        SetValTo1<<<BlockCount, 1024 >>>(dev_e2, m, n);

        cudaStreamSynchronize(0);

        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, 
        n, n, m,
        &alpha,
        dev_e1, m,
        dev_a, n,
        &beta,
        dev_c1, n);

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T, 
        m, n, n,
        &alpha,
        dev_e2, m,
        dev_c1, n,
        &beta,
        dev_res, m);

        cudaEventRecord(stop);

        cudaMemcpy(&reduce_sum, dev_res, sizeof(float), cudaMemcpyDeviceToHost);

        cudaEventSynchronize(stop);

        float GPUtime = 0;
        cudaEventElapsedTime(&GPUtime, start, stop);
        GPUtime *= 0.001;

        printf("Reduce elapsed time >>> %e sec\n", GPUtime);
        printf("Reduce sum is = %f\n", converter_back(reduce_sum) * max_val);


        cudaFree(dev_a);
        cudaFree(dev_e1);
        cudaFree(dev_c1);
        cudaFree(dev_e2);
        cudaFree(dev_res);

        free(host_a);

        return 0;
    }
#endif