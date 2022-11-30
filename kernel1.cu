
#include "common.h"
#include "timer.h"


#define THREADS_PER_BLOCK 1024
#define SM_FACTOR 5
#define SM_SIZE THREADS_PER_BLOCK*SM_FACTOR

__global__ void spmspm_gpu1_d(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, COOMatrix* cooMatrix) {

    // create shared memory of size SM_SIZE
    __shared__ float outputValues[SM_SIZE];

    // FIRST LOOP
    unsigned int ceilStride = (csrMatrix1->numCols+SM_SIZE-1)/SM_SIZE;
    for(unsigned int stride = 0; stride < ceilStride; ++stride){

        // initialize the shared memory to 0s
        for(int idx = 0; idx < SM_FACTOR; ++idx)
            outputValues[threadIdx.x+(idx*blockDim.x)] = 0.0f;
        __syncthreads();

        // SECOND LOOP through the rows
        for(unsigned int row_i = csrMatrix1->rowPtrs[blockIdx.x]; row_i < csrMatrix1->rowPtrs[blockIdx.x+1]; ++row_i) {

            unsigned int row = csrMatrix1->colIdxs[row_i];

            float theValue = csrMatrix1->values[row_i];

            unsigned int ceilStride2 = (csrMatrix1->rowPtrs[row+1]-csrMatrix1->rowPtrs[row]+blockDim.x-1)/blockDim.x;
            for(int stride2 = 0; stride2 < ceilStride2; ++stride2) {

                unsigned int i = csrMatrix1->rowPtrs[row] + threadIdx.x + (stride2*blockDim.x);

                if(i < csrMatrix1->rowPtrs[row+1]
                    && csrMatrix1->colIdxs[i] < (SM_SIZE*(stride+1))
                    && csrMatrix1->colIdxs[i] >= (SM_SIZE*stride)
                    ) {

                    unsigned int col = csrMatrix1->colIdxs[i];
                    float value = csrMatrix1->values[i];
                    outputValues[col-(SM_SIZE*stride)] += value*theValue;

                }

            }

            __syncthreads();

        }

        // write the outputValues to the COO matrix
        for(int stride2 = 0; stride2 < SM_FACTOR; ++stride2) {

            int outCol = threadIdx.x + (stride2*blockDim.x) + (SM_SIZE*stride);

            if(outputValues[threadIdx.x+(stride2*blockDim.x)] != 0) {

                atomicAdd(&cooMatrix->numNonzeros, 1);
                unsigned int j = cooMatrix->numNonzeros;
                cooMatrix->rowIdxs[j] = blockIdx.x;
                cooMatrix->colIdxs[j] = outCol;
                cooMatrix->values[j] = outputValues[threadIdx.x+(stride2*blockDim.x)];

            }

        }
        __syncthreads();
    }

}

void spmspm_gpu1(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, CSRMatrix* csrMatrix1_d, CSRMatrix* csrMatrix2_d, COOMatrix* cooMatrix_d, const GpuConfig& gpu_info) {


    // TODO what if one row was empty
    unsigned int numBlocks = csrMatrix1->numRows;
    unsigned int numThreadsPerBlock = THREADS_PER_BLOCK;

    spmspm_gpu1_d <<<numBlocks, numThreadsPerBlock>>> (csrMatrix1_d, csrMatrix2_d,  cooMatrix_d);


}

