#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

#define WARP_SIZE 32

#define TILE 16

using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main()
{
    // int N = 16;
    // int N_TILES = 256;
    int N_TOTAL = 512;

    printf("[+]   A: %d x %d\n", N_TOTAL, N_TOTAL);
	printf("[+]   B: %d x %d\n", N_TOTAL, N_TOTAL);
	printf("[*] Computing C = A * B  with using Cublas code...\n");

    for(int i=0;i<100;i++)
    {
        half *matrixA, *matrixB; 
        float *matrixC;

        matrixA = (half *)malloc(sizeof(half) * N_TOTAL * N_TOTAL);
        matrixB = (half *)malloc(sizeof(half) * N_TOTAL * N_TOTAL);
        matrixC = (float *)malloc(sizeof(float) * N_TOTAL * N_TOTAL);

        for (int i = 0; i < N_TOTAL*N_TOTAL; i++)
        {
            matrixA[i] = __float2half(rand() % 1000 / 100.0f);
            matrixB[i] = __float2half(rand() % 1000 / 100.0f);
            //matrixC[i] = 0.0f;
            matrixC[i] = rand() % 1000 / 100.0f;
        }

        half *d_matrixA, *d_matrixB;
        float *d_matrixC;

        cudaMalloc(&d_matrixA, N_TOTAL * N_TOTAL * sizeof(half));
        cudaMalloc(&d_matrixB, N_TOTAL * N_TOTAL * sizeof(half));
        cudaMalloc(&d_matrixC, N_TOTAL * N_TOTAL * sizeof(float));

        cudaMemcpy(d_matrixA, matrixA, N_TOTAL * N_TOTAL * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixB, matrixB, N_TOTAL * N_TOTAL * sizeof(half), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixC, matrixC, N_TOTAL * N_TOTAL * sizeof(float), cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);

        const float alpha = 1.0f;
        const float beta = 1.0f;

        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N_TOTAL, N_TOTAL, N_TOTAL,
            &alpha,
            d_matrixA, CUDA_R_16F, N_TOTAL,
            d_matrixB, CUDA_R_16F, N_TOTAL,
            &beta,
            d_matrixC, CUDA_R_32F, N_TOTAL,
            CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );

        cudaMemcpy(matrixC, d_matrixC, N_TOTAL * N_TOTAL * sizeof(float), cudaMemcpyDeviceToHost);

        cudaFree(d_matrixA);
        cudaFree(d_matrixB);
        cudaFree(d_matrixC);

    }
    
    // for(int i=0;i<N_TOTAL;i++)
    // {
    //     for(int j=0;j<N_TOTAL;j++)
    //     {
    //         printf("%.3f ",matrixC[i*N_TOTAL+j]);
    //     }
    //     printf("\n");
    // }
}