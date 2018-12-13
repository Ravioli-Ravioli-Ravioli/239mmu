#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>

#define BLOCK_WIDTH 16
#define TILE_WIDTH 16
#define width 2

//GlobalMem - From Kirk and Hwu, 2012, 
__global__ void matrixMulKernel(float* d_M, float* d_N, float* d_P, int Width) {
   // Calculate the row index of the d_Pelement and d_M
   int Row = blockIdx.y*blockDim.y+threadIdx.y;
   // Calculate the column index of d_P and d_N
   int Col = blockIdx.x*blockDim.x+threadIdx.x;

   if ((Row < Width) && (Col < Width)) {
      float Pvalue = 0;
      // each thread computes one element of the block sub-matrix
      for (int k = 0; k < Width; ++k) {
      Pvalue += d_M[Row*Width+k]*d_N[k*Width+Col];
      }
      d_P[Row*Width+Col] = Pvalue;
    }
}
//
__global__ void matrixMulKernel2(float* d_M, float* d_N, float* d_P,
int Width) {
   __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
   __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
   int bx = blockIdx.x; int by = blockIdx.y;
   int tx = threadIdx.x; int ty = threadIdx.y;
   // Identify the row and column of the d_P element to work on
   int Row = by * TILE_WIDTH + ty;
   int Col = bx * TILE_WIDTH + tx;
   float Pvalue = 0;
   // Loop over the d_M and d_N tiles required to compute d_P element
   for (int m = 0; m < Width/TILE_WIDTH; ++m) {
   // Coolaborative loading of d_M and d_N tiles into shared memory
      Mds[ty][tx] = d_M[Row*Width + m*TILE_WIDTH + tx];
      Nds[ty][tx] = d_N[(m*TILE_WIDTH + ty)*Width + Col];
      __syncthreads();
      for (int k = 0; k < TILE_WIDTH; ++k) {
        Pvalue += Mds[ty][k] * Nds[k][tx];
      }
      __syncthreads();
   }
   d_P[Row*Width + Col] = Pvalue;
}

//Main
int main(void){

//Print device properties
  int nDevices;
  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  MaxThreadPerBlock: %d\n",
           prop.maxThreadsPerBlock);
    printf("  MaxThreadsDim0: %d\n",
           prop.maxThreadsDim[0]);
    printf("  MaxThreadsDim1: %d\n",
           prop.maxThreadsDim[1]);
    printf("  MaxThreadsDim2: %d\n",
           prop.maxThreadsDim[2]);
    printf("  MaxGridSize: %d\n",
           prop.maxGridSize[1]);
    printf("  Warp Size: %d\n",
           prop.warpSize);
  }

//Allocate memory in host RAM
  float *A_h, *B_h, *C_h;
  cudaMallocHost((void **) &A_h, (width*width)*sizeof(float));
  cudaMallocHost((void **) &B_h, (width*width)*sizeof(float));
  cudaMallocHost((void **) &C_h, (width*width)*sizeof(float));
//Allocate memory in device RAM
  float *A_d, *B_d, *C_d;
  cudaMalloc((void **) &A_d, (width*width)*sizeof(float));
  cudaMalloc((void **) &B_d, (width*width)*sizeof(float));
  cudaMalloc((void **) &C_d, (width*width)*sizeof(float));
//Populate First Matrix
   int i, j; 
   srand(1);
   for (i = 0; i < width; i++){
      for (j = 0; j < width; j++) {
         A_h[i*width + j] = ((float)rand()/(float)(RAND_MAX)) * 100;
         printf("%.2f  ", A_h[i*width + j]);
      }
      printf("\n");
   }
   printf("\n");
//Populate Second Matrix
   for (i = 0; i < width; i++){
      for (j = 0; j < width; j++) {
         B_h[i*width + j] = ((float)rand()/(float)(RAND_MAX)) * 100;
	 printf("%.2f  ", B_h[i*width + j]);
      }
     printf("\n");
   }
//Mem copy from host to device
   cudaMemcpy(A_d, A_h, (width*width)*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(B_d, B_h, (width*width)*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(C_d, C_h, (width*width)*sizeof(float), cudaMemcpyHostToDevice);
//From Kirk and Hwu, 2012
   int NumBlocks = width/BLOCK_WIDTH;
   if (width % BLOCK_WIDTH) NumBlocks++;
   dim3 dimGrid(NumBlocks, NumBlocks);
   dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH);
//   matrixMulKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);
   matrixMulKernel2<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, width);
//Mem Copy
     cudaMemcpy(C_h, C_d, (width*width)*sizeof(float), cudaMemcpyDeviceToHost);
//Print matrix A
   for (i = 0; i < width; i++){
      for (j = 0; j < width; j++) {
         printf("%.2f  ", C_h[i*width + j]);
      }
      printf("\n");
   }
   printf("\n");
//Free up memory
   cudaFree(A_h);
   cudaFree(B_h);
   cudaFree(C_h);
   cudaFree(A_d); 
   cudaFree(B_d); 
   cudaFree(C_d);
}
