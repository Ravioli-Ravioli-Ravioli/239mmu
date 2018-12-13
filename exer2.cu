#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <time.h>

#define N 1000

////////////////////////////Each thread 1 row 1 column
__global__ void kernel_1t1e(float A[N][N], float B[N][N], float C[N][N], int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < size && j < size){
        A[i][j] = B[i][j] + C[i][j];
    }
}
///////////////////////////Each thread 1 row
__global__ void kernel_1t1r(float A[N][N], float B[N][N], float C[N][N], int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size){
	for (int j = 0; j < size; j++){
        A[i][j] = B[i][j] + C[i][j];
	}
    }
}
///////////////////////////Each thread 1 column
__global__ void kernel_1t1c(float A[N][N], float B[N][N], float C[N][N], int size) {
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (j < size){
        for (int i = 0; i < size; i++){
        A[i][j] = B[i][j] + C[i][j];
        }
    }
}

int main(void){
  int nDevices;
  int i, j;
  float A[N][N], B[N][N], C[N][N], (*A_d)[N], (*B_d)[N], (*C_d)[N];

//Print device properties
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
//Populate First Matrix
   srand(1);
   for (i = 0; i < N; i++){
      for (j = 0; j < N; j++) {
         B[i][j] = ((float)rand()/(float)(RAND_MAX)) * 100;
//         printf("%f  ", B[i][j]);
      }
      printf("\n");
   }
   printf("\n");
//Populate Second Matrix
   for (i = 0; i < N; i++){
      for (j = 0; j < N; j++) {
         C[i][j] = ((float)rand()/(float)(RAND_MAX)) * 100;
//	 printf("%f  ", C[i][j]);
      }
     printf("\n");
   }
   printf("\n");
   printf("===============================");
   printf("\n");

//Allocate memory in the device
   
   cudaMalloc((void**) &A_d, (N*N)*sizeof(float));
   cudaMalloc((void**) &B_d, (N*N)*sizeof(float));
   cudaMalloc((void**) &C_d, (N*N)*sizeof(float));

//Mem copy from host to device
   cudaMemcpy(A_d, A, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(B_d, B, (N*N)*sizeof(float), cudaMemcpyHostToDevice);
   cudaMemcpy(C_d, C, (N*N)*sizeof(float), cudaMemcpyHostToDevice);

   dim3 threadsPerBlock(N, N);
   dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

   cudaEvent_t start, stop;
   float elapsed = 0;

//ThreadAll

//Run
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   kernel_1t1e<<<numBlocks,threadsPerBlock>>>(A_d, B_d, C_d, N);
   cudaEventRecord(stop, 0);

   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("GPU Run TIme threadsall %.2f ms \n", elapsed);

////////////////////////////////////Thread Row
/*
   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   kernel_1t1r<<<numBlocks,threadsPerBlock>>>(A_d, B_d, C_d, N);
   cudaEventRecord(stop, 0);

   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("GPU Run TIme threadsrow %.2f ms \n", elapsed);

////////////////////////////////////Thread Column

   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);
   kernel_1t1c<<<numBlocks,threadsPerBlock>>>(A_d, B_d, C_d, N);
   cudaEventRecord(stop, 0);

   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&elapsed, start, stop);
   cudaEventDestroy(start);
   cudaEventDestroy(stop);
   printf("GPU Run TIme threadscol %.2f ms \n", elapsed);
*/
//Mem Copy
   cudaMemcpy(A, A_d, (N*N)*sizeof(float), cudaMemcpyDeviceToHost);

/////////////////////////////////////Print matrix A

   for (i = 0; i < N; i++){
      for (j = 0; j < N; j++) {
         printf("%f  ", A[i][j]);
      }
      printf("\n");
   }
   printf("\n");
	
/////////////////////////////////////Free up memory
   cudaFree(A_d); 
   cudaFree(B_d); 
   cudaFree(C_d);
}
