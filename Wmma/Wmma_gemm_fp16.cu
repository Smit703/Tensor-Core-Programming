/* Computing C = A*B + C using naive cuda code */

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
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

__global__ void WMMA_Multiplication(half *A, half* B, float*C, int N, int N_TOTAL)
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x)/WARP_SIZE;
	int iy = (blockIdx.y * blockDim.y + threadIdx.y);
	
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, TILE, TILE, TILE, half, nvcuda::wmma::row_major> a_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, TILE, TILE, TILE, half, nvcuda::wmma::col_major> b_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE, TILE, TILE, float> ab_frag;
	nvcuda::wmma::fragment<nvcuda::wmma::accumulator, TILE, TILE, TILE, float> c_frag;
	
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


// __shared__ float As[TILE][TILE];
// __shared__ float Bs[TILE][TILE];
// __shared__ float Cs[TILE][TILE];

// int bx = blockIdx.x;
// int by = blockIdx.y;
// int tx = threadIdx.x;
// int ty = threadIdx.y;

// // Define matrix fragments
// fragment<matrix_a, TILE, TILE, TILE, half, row_major> a_frag;
// fragment<matrix_b, TILE, TILE, TILE, half, col_major> b_frag;
// fragment<accumulator, TILE, TILE, TILE, float> c_frag;

// // Initialize fragments
// load_matrix_sync(a_frag, A + (by * blockDim.y + ty) * N + (tx + bx * blockDim.x), N);
// load_matrix_sync(b_frag, B + (by * blockDim.y + ty) * N + (tx + bx * blockDim.x), N);
// load_matrix_sync(c_frag, C + (by * blockDim.y + ty) * N + (tx + bx * blockDim.x), N, mem_row_major);

// // MMA operation
// mma_sync(c_frag, a_frag, b_frag, c_frag);

// // Store the result to global memory
// store_matrix_sync(C + (by * blockDim.y + ty) * N + (tx + bx * blockDim.x), c_frag, N, mem_row_major);



//__global__ void WMMA_Multiplication(half *A, half* B, float*C, int N, int N_TOTAL)
// {
//    int lda = N;
//    int ldb = N;
//    int ldc = N;

//    // Tile using a 2D grid
//    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
//    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
 
//    // Declare the fragments
//    wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, half, wmma::col_major> a_frag;
//    wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, half, wmma::col_major> b_frag;
//    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> acc_frag;
//    wmma::fragment<wmma::accumulator, TILE, TILE, TILE, float> c_frag;

//    wmma::fill_fragment(acc_frag, 0.0f);

//    // Loop over k
//    for (int i = 0; i < N_TOTAL; i += N) {
//       int aRow = warpM * N;
//       int aCol = i;

//       int bRow = i;
//       int bCol = warpN * N;

//       // Bounds checking
//       if (aRow < N_TOTAL && aCol < N_TOTAL && bRow < N_TOTAL && bCol < N_TOTAL) {
//          // Load the inputs
//          wmma::load_matrix_sync(a_frag, A + aRow + aCol * lda, lda);
//          wmma::load_matrix_sync(b_frag, B + bRow + bCol * ldb, ldb);

//          // Perform the matrix multiplication
//          wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
//       }
//    }

//    // Load in the current value of c, scale it by beta, and add this our result scaled by alpha
//    int cRow = warpM * N;
//    int cCol = warpN * N;

//    if (cRow < N_TOTAL && cCol < N_TOTAL) {
//       wmma::load_matrix_sync(c_frag, C + cRow + cCol * ldc, ldc, wmma::mem_col_major);

// #pragma unroll
//       for(int i=0; i < c_frag.num_elements; i++) {
//          c_frag.x[i] = acc_frag.x[i] + c_frag.x[i];
//       }

//       // Store the output
//       wmma::store_matrix_sync(C + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
//    }
// }