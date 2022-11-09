#include "common.h"
#include "timer.h"

#define THREADS_PER_BLOCK 1024
#define BLOCK_REUSE 300

__global__ void spmspm_gpu0_d(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, COOMatrix* cooMatrix, unsigned int numBlocks, float * blockArrays){    
    unsigned int numCols2 = csrMatrix2->numCols; // nuumber of coloumns in the second matrix
    // initialize array to 0's
    for (int y = threadIdx.x; y < numCols2; y += blockDim.x){ // initialize shared array to be 0
        blockArrays[blockIdx.x*numCols2 + threadIdx.x] = 0.0f;
    }
    __syncthreads();

    
    for(unsigned int i = 0; i < BLOCK_REUSE;++i){ // block reuse
        unsigned int row1 = blockIdx.x + numBlocks*i; // current row the block is representing
        if(row1 < csrMatrix1->numRows){ // make sure block is representing a row
            for(unsigned int rowPtr=csrMatrix1->rowPtrs[row1]; rowPtr < csrMatrix1->rowPtrs[row1+1]; ++rowPtr){ // go through the non-zero's of a row
                unsigned int col1 = csrMatrix1->colIdxs[rowPtr]; // column of current element
                float value = csrMatrix1->values[rowPtr]; // value of current element
                unsigned int row2 = col1; // row to be used in second matrix
                
                unsigned int numElementsRow = csrMatrix2->rowPtrs[row2+1] - csrMatrix2->rowPtrs[row2]; // number of non-zero elements in the second matrix on the specified row
                for(int j = threadIdx.x; j < numElementsRow; j += blockDim.x){ // assign all the threads to elements in the second array
                    unsigned int rowPtr2 = csrMatrix2->rowPtrs[row2] + j;
                    unsigned int col2 = csrMatrix2->colIdxs[rowPtr2];
                    float value2 = csrMatrix2->values[rowPtr2];
                    blockArrays[blockIdx.x*numCols2 + col2] += value*value2;
                }
                __syncthreads(); // wait for all threads to finish before going to next non-zero in matrix 1
            }

            for(int elemCol = threadIdx.x; elemCol < numCols2; elemCol += blockDim.x) {
                if(blockArrays[blockIdx.x*numCols2 + elemCol] != 0) {
                    unsigned int k = atomicAdd(&cooMatrix->numNonzeros, 1);
                    float valueTemp = blockArrays[blockIdx.x*numCols2 + elemCol];
                    
                    //add to coo matrix
                    cooMatrix->rowIdxs[k] = row1;
                    cooMatrix->colIdxs[k] = elemCol;
                    cooMatrix->values[k] = valueTemp;

                    blockArrays[blockIdx.x*numCols2 + elemCol] = 0.0f; // reset the value for next row
                    if(threadIdx.x==0 && row1==0 && elemCol == 0) printf("%d, %f\n",k,valueTemp);
                }
            }
        }

        __syncthreads(); // all threads must finish before going to the next block
    }
}

void spmspm_gpu0(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d) {
    static unsigned int numThreadsPerBlock = THREADS_PER_BLOCK;
    static unsigned int numBlocks = (csrMatrix1->numRows + BLOCK_REUSE - 1) / BLOCK_REUSE;

    // create a row for every block
    float* blockArrays_d;
    cudaMalloc((void**) &blockArrays_d, numBlocks * csrMatrix2->numCols * sizeof(float));

    spmspm_gpu0_d <<<numBlocks, numThreadsPerBlock>>> (csrMatrix1_d, csrMatrix2_d, cooMatrix_d, numBlocks, blockArrays_d);

    cudaFree(blockArrays_d);
}