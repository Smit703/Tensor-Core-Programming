/* Computing C = A*B + C using naive cuda code */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#define WARP_SIZE 32

// Defined in main
// // MMA matrix tile dimensions.
// #define N 16

// // GEMM configuration.
// #define N_TILES 256

// //Dimensions
// #define N_TOTAL (N * N_TILES)

#define TILE 16

using namespace std;
using namespace nvcuda;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void WMMA_Multiplication(signed char *A, signed char *B, int *C, int N, int N_TOTAL)
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x)/WARP_SIZE;
	int iy = (blockIdx.y * blockDim.y + threadIdx.y);
	
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE, TILE, TILE, signed char, nvcuda::wmma::row_major> a_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE, TILE, TILE, signed char, nvcuda::wmma::col_major> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE, TILE, TILE, int> ab_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE, TILE, TILE, int> c_frag;
	
	nvcuda::wmma::fill_fragment(ab_frag, 0.0f);

	// AB = A*B
	int a_col, a_row, b_col, b_row, c_col, c_row;
	a_row = ix * N;
	b_row = iy * N;
	for (int k=0; k<N_TOTAL; k+=N) {
		a_col = b_col = k;

		if (a_row < N_TOTAL && a_col < N_TOTAL && b_row < N_TOTAL && b_col < N_TOTAL) {
			// Load the inputs
			nvcuda::wmma::load_matrix_sync(a_frag, A + a_col + a_row * N_TOTAL, N_TOTAL);
			nvcuda::wmma::load_matrix_sync(b_frag, B + b_col + b_col * N_TOTAL, N_TOTAL);

			// Perform the matrix multiplication
			nvcuda::wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
		}
	}

	// D = AB + C
	c_col = b_row;
	c_row = a_row;
	if (c_row < N_TOTAL && c_col < N_TOTAL) {
		nvcuda::wmma::load_matrix_sync(c_frag, C + c_col + c_row * N_TOTAL, N_TOTAL, nvcuda::wmma::mem_row_major);

		for (int i = 0; i < c_frag.num_elements; i++) {
			c_frag.x[i] = ab_frag.x[i] + c_frag.x[i];
		}

		// Store the output
		nvcuda::wmma::store_matrix_sync(C + c_col + c_row * N_TOTAL, c_frag, N_TOTAL, nvcuda::wmma::mem_row_major);
	}
    
}


int main()
{
    int N = 16;
    int N_TILES = 256;
    int N_TOTAL = N * N_TILES;

    printf("[+]   A: %d x %d\n", N_TOTAL, N_TOTAL);
	printf("[+]   B: %d x %d\n", N_TOTAL, N_TOTAL);
	printf("[*] Computing C = A * B  with using WMMA code...\n");

	for(int i=0;i<100;i++)
	{
		signed char *matrixA, *matrixB; 
		int *matrixC;

		matrixA = (signed char *)malloc(sizeof(signed char) * N_TOTAL * N_TOTAL);
		matrixB = (signed char *)malloc(sizeof(signed char) * N_TOTAL * N_TOTAL);
		matrixC = (int *)malloc(sizeof(int) * N_TOTAL * N_TOTAL);

		for (int i = 0; i < N_TOTAL*N_TOTAL; i++)
		{
			matrixA[i] = (rand() % 256) - 128;
            matrixB[i] = (rand() % 256) - 128;
            //matrixC[i] = 0.0f;
            matrixC[i] = rand() % 4096;
		}

		signed char *d_matrixA, *d_matrixB;
		int *d_matrixC;

		cudaMalloc(&d_matrixA, N_TOTAL * N_TOTAL * sizeof(signed char));
		cudaMalloc(&d_matrixB, N_TOTAL * N_TOTAL * sizeof(signed char));
		cudaMalloc(&d_matrixC, N_TOTAL * N_TOTAL * sizeof(int));

		cudaMemcpy(d_matrixA, matrixA, N_TOTAL * N_TOTAL * sizeof(signed char), cudaMemcpyHostToDevice);
		cudaMemcpy(d_matrixB, matrixB, N_TOTAL * N_TOTAL * sizeof(signed char), cudaMemcpyHostToDevice);
		cudaMemcpy(d_matrixC, matrixC, N_TOTAL * N_TOTAL * sizeof(int), cudaMemcpyHostToDevice);

		int gridDimx, gridDimy;
		dim3 block(128,4,1); 
		
		gridDimx = (N_TOTAL + (N * block.x / WARP_SIZE - 1)) / (N * block.x / WARP_SIZE);
		gridDimy = (N_TOTAL + N * block.y - 1) / (N * block.y);

		dim3 grid(gridDimx,gridDimy,1);

		// gridDimx = (N_TOTAL/128) + (N_TOTAL%128!=0);
		// gridDimy = (N_TOTAL/4) + (N_TOTAL%4!=0);
		// dim3 grid2(gridDimx,gridDimy,1);
		// dim3 block2(128,4,1);

		WMMA_Multiplication<<<grid,block>>>(d_matrixA,d_matrixB,d_matrixC,N,N_TOTAL);
		gpuErrchk( cudaDeviceSynchronize() );

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
