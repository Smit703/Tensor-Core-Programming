#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
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
    int N_TOTAL = 2047;

    printf("[+]   A: %d x %d\n", N_TOTAL, N_TOTAL);
	printf("[+]   B: %d x %d\n", N_TOTAL, N_TOTAL);
	printf("[*] Computing C = A * B  with using Cublas code...\n");

    for(int i=0;i<100;i++)
    {
        int8_t *matrixA, *matrixB; 
        int *matrixC;

        matrixA = (int8_t *)malloc(sizeof(int8_t) * N_TOTAL * N_TOTAL);
        matrixB = (int8_t *)malloc(sizeof(int8_t) * N_TOTAL * N_TOTAL);
        matrixC = (int *)malloc(sizeof(int) * N_TOTAL * N_TOTAL);

        for (int i = 0; i < N_TOTAL*N_TOTAL; i++)
        {
            matrixA[i] = (rand() % 256) - 128;
            matrixB[i] = (rand() % 256) - 128;
            //matrixC[i] = 0.0f;
            matrixC[i] = rand() % 4096;
        }

        int8_t *d_matrixA, *d_matrixB;
        int *d_matrixC;

        cudaMalloc(&d_matrixA, N_TOTAL * N_TOTAL * sizeof(int8_t));
        cudaMalloc(&d_matrixB, N_TOTAL * N_TOTAL * sizeof(int8_t));
        cudaMalloc(&d_matrixC, N_TOTAL * N_TOTAL * sizeof(int));

        cudaMemcpy(d_matrixA, matrixA, N_TOTAL * N_TOTAL * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixB, matrixB, N_TOTAL * N_TOTAL * sizeof(int8_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrixC, matrixC, N_TOTAL * N_TOTAL * sizeof(int), cudaMemcpyHostToDevice);

        cublasHandle_t handle;
        cublasCreate(&handle);

        const int alpha = 1;
        const int beta = 1;

        cublasGemmEx(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            N_TOTAL, N_TOTAL, N_TOTAL,
            &alpha,
            d_matrixA, CUDA_R_8I, N_TOTAL,
            d_matrixB, CUDA_R_8I, N_TOTAL,
            &beta,
            d_matrixC, CUDA_R_32I, N_TOTAL,
            CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP
        );

        cudaMemcpy(matrixC, d_matrixC, N_TOTAL * N_TOTAL * sizeof(int), cudaMemcpyDeviceToHost);

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