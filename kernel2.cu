#include "common.h"
#include "timer.h"

#define THREADS_PER_BLOCK 1024

__device__ unsigned int counter_global = 0;

__global__ void spmspm_gpu2_d(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, COOMatrix* cooMatrix, float * blockArrays, float * columnArrays){
    unsigned int numCols2 = csrMatrix2->numCols; // nuumber of coloumns in the second matrix
    __shared__ unsigned int row1;
    // initialize the counter
    __shared__ unsigned int counter;
    
    // initialize array to 0's
    for (int y = threadIdx.x; y < numCols2; y += blockDim.x){ // initialize shared array to be 0
        blockArrays[blockIdx.x*numCols2 + threadIdx.x] = 0.0f;
        columnArrays[blockIdx.x*numCols2 + threadIdx.x] = 0.0f;
    }

    if(threadIdx.x == 0){
        row1 = atomicAdd(&counter_global, 1);
        counter = 0;
    }
    __syncthreads();

    while(row1 < csrMatrix1->numRows){
        for(unsigned int rowPtr=csrMatrix1->rowPtrs[row1]; rowPtr < csrMatrix1->rowPtrs[row1+1]; ++rowPtr){ // go through the non-zero's of a row
            
            unsigned int col1 = csrMatrix1->colIdxs[rowPtr]; // column of current element
            float value = csrMatrix1->values[rowPtr]; // value of current element
            
            unsigned int row2 = col1; // row to be used in second matrix
            
            unsigned int numElementsRow = csrMatrix2->rowPtrs[row2+1] - csrMatrix2->rowPtrs[row2]; // number of non-zero elements in the second matrix on the specified row
            for(int j = threadIdx.x; j < numElementsRow; j += blockDim.x){ // assign all the threads to elements in the second array
                unsigned int rowPtr2 = csrMatrix2->rowPtrs[row2] + j;
                unsigned int col2 = csrMatrix2->colIdxs[rowPtr2];
                float value2 = csrMatrix2->values[rowPtr2];
                float oldVal = blockArrays[blockIdx.x*numCols2 + col2];
                blockArrays[blockIdx.x*numCols2 + col2] += value*value2;
                if (oldVal == 0.0f) {
                    columnArrays[blockIdx.x*numCols2 + atomicAdd(&counter, 1)] = col2;
                }

            }
            __syncthreads(); // wait for all threads to finish before going to next non-zero in matrix 1
        }

        for(int i = threadIdx.x; i < counter; i += blockDim.x) {

            unsigned int col = columnArrays[blockIdx.x*numCols2 + i];
            float value = blockArrays[blockIdx.x*numCols2 + col];
            unsigned int k = atomicAdd(&cooMatrix->numNonzeros, 1);

            //add to coo matrix
            cooMatrix->rowIdxs[k] = row1;
            cooMatrix->colIdxs[k] = col;
            cooMatrix->values[k] = value;

            blockArrays[blockIdx.x*numCols2 + col] = 0.0f; // reset the value for next row
        }
        __syncthreads();

        if(threadIdx.x == 0){
            row1 = atomicAdd(&counter_global, 1);
            counter = 0;
        }
        __syncthreads(); // all threads must finish before going to the next block
    }
}

void spmspm_gpu2(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d, const GpuConfig& gpu_info) {
    static unsigned int numThreadsPerBlock = THREADS_PER_BLOCK;
    int numSM = gpu_info.MultiprocessorCount();
    static unsigned int numBlocks = 2*numSM;
    
    // create a row for every block
    float* blockArrays_d;
    cudaMalloc((void**) &blockArrays_d, numBlocks * csrMatrix2->numCols * sizeof(float));
    float* columnArrays_d;
    cudaMalloc((void**) &columnArrays_d, numBlocks * csrMatrix2->numCols * sizeof(float));

    spmspm_gpu2_d <<<numBlocks, numThreadsPerBlock>>> (csrMatrix1_d, csrMatrix2_d, cooMatrix_d, blockArrays_d, columnArrays_d);

    cudaFree(blockArrays_d);
}
