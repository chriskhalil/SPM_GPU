
#include "common.h"
#include "timer.h"

#define THREADS_PER_BLOCK 1024
#define BLOCK_REUSE 300
#define SHARED_MULTIPLIER 10
#define SHARED_SIZE SHARED_MULTIPLIER*THREADS_PER_BLOCK // can go up to 24,000 per SM on a volta v100

__global__ void spmspm_gpu1_d(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, COOMatrix* cooMatrix, unsigned int numBlocks){    

    unsigned int numCols2 = csrMatrix2->numCols; // number of coloumns in the second matrix
    __shared__ float outputValues[SHARED_SIZE]; // used to store running sums

    //initialize shared memory to 0
    for(unsigned int x = threadIdx.x; x < SHARED_SIZE; x += blockIdx.x){
        outputValues[x] = 0.0f;
    }


    for(unsigned int i = 0; i < BLOCK_REUSE;++i){ // block reuse

        unsigned int row1 = blockIdx.x + numBlocks*i; // current row the block is representing

        if(row1 < csrMatrix1->numRows){ // make sure block is representing a row

            unsigned int numSections = (numCols2 + SHARED_SIZE -1)/SHARED_SIZE;
            for(int section = 0; section < numSections; ++section){ // split the second matrix into sections

                for(unsigned int rowPtr=csrMatrix1->rowPtrs[row1]; rowPtr < csrMatrix1->rowPtrs[row1+1]; ++rowPtr){ // go through the non-zero's of a row

                    unsigned int col1 = csrMatrix1->colIdxs[rowPtr]; // column of current element
                    float value = csrMatrix1->values[rowPtr]; // value of current element

                    unsigned int row2 = col1; // row to be used in second matrix

                    unsigned int numElementsRow = csrMatrix2->rowPtrs[row2+1] - csrMatrix2->rowPtrs[row2]; // number of non-zero elements in the second matrix on the specified row
                    
                    for(int j = threadIdx.x; j < SHARED_SIZE && j < numElementsRow; j += blockDim.x){ // assign all the threads to elements in the second array
                        unsigned int rowPtr2 = csrMatrix2->rowPtrs[row2] + j + section*SHARED_SIZE;
                        unsigned int col2 = csrMatrix2->colIdxs[rowPtr2];
                        float value2 = csrMatrix2->values[rowPtr2];
                        outputValues[col2 % SHARED_SIZE] += value*value2;
                    }
                    __syncthreads(); // wait for all threads to finish before going to next non-zero in matrix 1
                }
                __syncthreads(); // wait for all the threads to finish calculating for the section

                //copy from shared memory to COO
                for(int elemCol = threadIdx.x; elemCol < SHARED_SIZE && elemCol + section*SHARED_SIZE < numCols2; elemCol += blockDim.x) {
                    if(outputValues[elemCol] != 0) {
                        unsigned int k = atomicAdd(&cooMatrix->numNonzeros, 1);
                        float valueTemp = outputValues[blockIdx.x*elemCol];
                        
                        //add to coo matrix
                        cooMatrix->rowIdxs[k] = row1;
                        cooMatrix->colIdxs[k] = elemCol + section*SHARED_SIZE;
                        cooMatrix->values[k] = valueTemp;

                        outputValues[elemCol] = 0.0f; // reset the value for next row
                    }
                }
            }

        }
        __syncthreads(); // all threads must finish before going to the next block

    }
}


void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d, const GpuConfig& gpu_info) {

    static unsigned int numThreadsPerBlock = THREADS_PER_BLOCK;
    static unsigned int numBlocks = (csrMatrix1->numRows + BLOCK_REUSE - 1) / BLOCK_REUSE;

    spmspm_gpu1_d <<<numBlocks, numThreadsPerBlock>>> (csrMatrix1_d, csrMatrix2_d, cooMatrix_d, numBlocks);
}

