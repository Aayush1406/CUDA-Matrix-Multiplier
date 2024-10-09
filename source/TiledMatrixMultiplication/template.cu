#include <cuda_runtime.h> 
#include <device_launch_parameters.h> 
#include <wb.h>

#define TILE_WIDTH 16 	//do not change this value

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
// Compute C = A * B
// Compute C = A * B
__global__ void matrixMultiplyShared(float* A, float* B, float* C, int numARows, int numAColumns, int numBColumns) {
    //@@ Insert code to implement tiled matrix multiplication here
    //@@ You have to use shared memory to write this kernel

    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y, tx = threadIdx.x, ty = threadIdx.y;

    int row = blockDim.y * by + ty;
    int col = blockDim.x * bx + tx;

    float pvalue = 0;
    
    for (int p = 0; p < (numAColumns -1 )/ TILE_WIDTH + 1; p++) { // Will run 128 times

        if (row < numARows && p * TILE_WIDTH + tx < numAColumns) {

            ds_A[ty][tx] = A[row * numAColumns + p * TILE_WIDTH + tx];
        }
        else {
            ds_A[ty][tx] = 0.0;

        }


        if (col < numBColumns && p * TILE_WIDTH + ty < numAColumns) {

           ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * numBColumns + col];

        }
        else {

            ds_B[ty][tx] = 0.0;

        }

        __syncthreads();
        for (int i = 0; i < TILE_WIDTH; i++) {

        pvalue = pvalue + ds_A[ty][i] * ds_B[i][tx];

        __syncthreads();
        }

    }

    if (row < numARows && col < numBColumns) {

        C[row * numBColumns + col] = pvalue;

    }
}

//__global__ void matrixMultiplyShared(float* A, float* B, float* C,
//    int numARows, int numAColumns,
//    int numBColumns) {
//    //@@ Insert code to implement tiled matrix multiplication here
//    //@@ You have to use shared memory to write this kernel
//    __shared__ float Ashared[TILE_WIDTH][TILE_WIDTH];
//    __shared__ float Bshared[TILE_WIDTH][TILE_WIDTH];
//
//    int bx = blockIdx.x;
//    int by = blockIdx.y;
//    int tx = threadIdx.x;
//    int ty = threadIdx.y;
//
//    int Row = (by * blockDim.y) + ty;
//    int Col = (bx * blockDim.x) + tx;
//
//    float Pvalue = 0;
//
//    for (int p = 0; p < (numAColumns - 1) / TILE_WIDTH + 1; p++) {
//        if (Row < numARows && p * TILE_WIDTH + tx < numAColumns) {
//            Ashared[ty][tx] = A[Row * numAColumns + (p * TILE_WIDTH + tx)];
//        }
//        else {
//            Ashared[ty][tx] = 0.0;
//        }
//        if (p * TILE_WIDTH + ty < numAColumns && Col < numBColumns) {
//            Bshared[ty][tx] = B[(p * TILE_WIDTH + ty) * numBColumns + Col];
//        }
//        else {
//            Bshared[ty][tx] = 0.0;
//        }
//        __syncthreads();
//
//        for (int i = 0; i < TILE_WIDTH; i++) {
//            Pvalue += Ashared[ty][i] * Bshared[i][tx];
//        }
//        __syncthreads();
//
//
//    }

//if (Row < numARows && Col < numBColumns) {
//    C[Row * numBColumns + Col] = Pvalue;
//}
//
//}

int main(int argc, char** argv) {
    wbArg_t args;
    float* hostA; // The A matrix
    float* hostB; // The B matrix
    float* hostC; // The output C matrix
    float* deviceA;
    float* deviceB;
    float* deviceC;
    int numARows;    // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows;    // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows;    // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set
                     // this)

    hostC = NULL;

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float*)wbImport(wbArg_getInputFile(args, 0), &numARows,
        &numAColumns);
    hostB = (float*)wbImport(wbArg_getInputFile(args, 1), &numBRows,
        &numBColumns);
    //@@ Set numCRows and numCColumns
    numCRows = numARows;      // Rows of C = Rows of A
    numCColumns = numBColumns; // Columns of C = Columns of B

    //@@ Allocate the hostC matrix
    hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));
    wbTime_stop(Generic, "Importing data and creating memory on host");

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here
    wbCheck(cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float)));
    wbCheck(cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float)));

    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here
    wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
    wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));

    wbTime_stop(GPU, "Copying input memory to the GPU.");

    //@@ Initialize the grid and block dimensions here
    //dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    //dim3 dimGrid((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH, (numCRows + TILE_WIDTH - 1) / TILE_WIDTH);
    dim3 dimGrid((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
    matrixMultiplyShared << <dimGrid, dimBlock >> > (deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns);

    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
    wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here
    wbCheck(cudaFree(deviceA));
    wbCheck(cudaFree(deviceB));
    wbCheck(cudaFree(deviceC));

    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}
