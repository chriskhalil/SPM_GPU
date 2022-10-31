
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "common.h"
#include "matrix.h"
#include "timer.h"

void spmspm_cpu(CSRMatrix* csrMatrix1, CSRMatrix* csrMatrix2, COOMatrix* cooMatrix) {
    float* outputValues = (float*) malloc(csrMatrix2->numCols*sizeof(float));
    memset(outputValues, 0, csrMatrix2->numCols*sizeof(float));
    unsigned int* outputCols = (unsigned int*) malloc(csrMatrix2->numCols*sizeof(unsigned int));
    unsigned int numOutputCols = 0;
    for(unsigned int row1 = 0; row1 < csrMatrix1->numRows; ++row1) {
        for(unsigned int i1 = csrMatrix1->rowPtrs[row1]; i1 < csrMatrix1->rowPtrs[row1 + 1]; ++i1) {
            unsigned int col1 = csrMatrix1->colIdxs[i1];
            float value1 = csrMatrix1->values[i1];
            unsigned int row2 = col1;
            for(unsigned int i2 = csrMatrix2->rowPtrs[row2]; i2 < csrMatrix2->rowPtrs[row2 + 1]; ++i2) {
                unsigned int col2 = csrMatrix2->colIdxs[i2];
                float value2 = csrMatrix2->values[i2];
                float oldVal = outputValues[col2];
                outputValues[col2] += value1*value2;
                if(oldVal == 0.0f) { // Assuming all matrix entries are positive, so oldVal cannot become 0 again
                    outputCols[numOutputCols++] = col2;
                }
            }
        }
        for(unsigned int i = 0; i < numOutputCols; ++i) {
            unsigned int col = outputCols[i];
            float value = outputValues[col];
            outputValues[col] = 0.0f;
            assert(cooMatrix->numNonzeros < cooMatrix->capacity);
            unsigned int j = cooMatrix->numNonzeros++;
            cooMatrix->rowIdxs[j] = row1;
            cooMatrix->colIdxs[j] = col;
            cooMatrix->values[j] = value;
        }
        numOutputCols = 0;
    }
    free(outputValues);
    free(outputCols);
}

void verify(COOMatrix* cooMatrixGPU, COOMatrix* cooMatrixCPU, unsigned int quickVerify) {
    if(cooMatrixCPU->numNonzeros != cooMatrixGPU->numNonzeros) {
        printf("    \033[1;31mMismatching number of non-zeros (CPU result = %d, GPU result = %d)\033[0m\n", cooMatrixCPU->numNonzeros, cooMatrixGPU->numNonzeros);
        return;
    } else if(quickVerify) {
        printf("    Quick verification succeeded\n");
        printf("        This verification is not exact. For exact verification, pass the -v flag.\n");
    } else {
        printf("    Verifying result\n");
        sortCOOMatrix(cooMatrixCPU);
        sortCOOMatrix(cooMatrixGPU);
        for(unsigned int i = 0; i < cooMatrixCPU->numNonzeros; ++i) {
            unsigned int rowCPU = cooMatrixCPU->rowIdxs[i];
            unsigned int rowGPU = cooMatrixGPU->rowIdxs[i];
            unsigned int colCPU = cooMatrixCPU->colIdxs[i];
            unsigned int colGPU = cooMatrixGPU->colIdxs[i];
            float valCPU = cooMatrixCPU->values[i];
            float valGPU = cooMatrixGPU->values[i];
            if(rowCPU != rowGPU || colCPU != colGPU || abs(valGPU - valCPU)/valCPU > 1e-5) {
                printf("        \033[1;31mMismatch detected: CPU: (%d, %d, %f), GPU: (%d, %d, %f)\033[0m\n", rowCPU, colCPU, valCPU, rowGPU, colGPU, valGPU);
                return;
            }
        }
        printf("        Verification succeeded\n");
    }
}

int main(int argc, char**argv) {

    cudaDeviceSynchronize();
    setbuf(stdout, NULL);

    // Parse arguments
    const char* matrixFile = "data/matrix0.txt";
    unsigned int runGPUVersion0 = 0;
    unsigned int runGPUVersion1 = 0;
    unsigned int runGPUVersion2 = 0;
    unsigned int runGPUVersion3 = 0;
    unsigned int runGPUVersion4 = 0;
    unsigned int quickVerify = 1;
    int opt;
    while((opt = getopt(argc, argv, "f:01234v")) >= 0) {
        switch(opt) {
            case 'f': matrixFile = optarg;  break;
            case '0': runGPUVersion0 = 1;   break;
            case '1': runGPUVersion1 = 1;   break;
            case '2': runGPUVersion2 = 1;   break;
            case '3': runGPUVersion3 = 1;   break;
            case '4': runGPUVersion4 = 1;   break;
            case 'v': quickVerify = 0;      break;
            default:  fprintf(stderr, "\nUnrecognized option!\n");
                      exit(0);
        }
    }

    // Allocate memory and initialize data
    printf("Reading matrix from file: %s\n", matrixFile);
    CSRMatrix* csrMatrix = createCSRMatrixFromFile(matrixFile);
    printf("Allocating COO matrices\n");
    COOMatrix* cooMatrix = createEmptyCOOMatrix(csrMatrix->numRows, csrMatrix->numRows, csrMatrix->numRows*10000);
    COOMatrix* cooMatrix_h = createEmptyCOOMatrix(csrMatrix->numRows, csrMatrix->numRows, csrMatrix->numRows*10000);

    // Compute on CPU
    printf("Running CPU version\n");
    Timer timer;
    startTime(&timer);
    spmspm_cpu(csrMatrix, csrMatrix, cooMatrix);
    stopTime(&timer);
    printElapsedTime(timer, "    CPU time", CYAN);

    if(runGPUVersion0 || runGPUVersion1 || runGPUVersion2 || runGPUVersion3 || runGPUVersion4) {

        // Allocate GPU memory
        startTime(&timer);
        CSRMatrix* csrMatrix_d = createEmptyCSRMatrixOnGPU(csrMatrix->numRows, csrMatrix->numCols, csrMatrix->numNonzeros);
        COOMatrix* cooMatrix_d = createEmptyCOOMatrixOnGPU(cooMatrix->numRows, cooMatrix->numCols, cooMatrix->capacity);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "GPU allocation time");

        // Copy data to GPU
        startTime(&timer);
        copyCSRMatrixToGPU(csrMatrix, csrMatrix_d);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "Copy to GPU time");

        if(runGPUVersion0) {

            printf("Running GPU version 0\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 0
            startTime(&timer);
            spmspm_gpu0(csrMatrix, csrMatrix, csrMatrix_d, csrMatrix_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 0)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        if(runGPUVersion1) {

            printf("Running GPU version 1\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 1
            startTime(&timer);
            spmspm_gpu1(csrMatrix, csrMatrix, csrMatrix_d, csrMatrix_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 1)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        if(runGPUVersion2) {

            printf("Running GPU version 2\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 2
            startTime(&timer);
            spmspm_gpu2(csrMatrix, csrMatrix, csrMatrix_d, csrMatrix_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 2)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        if(runGPUVersion3) {

            printf("Running GPU version 3\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 3
            startTime(&timer);
            spmspm_gpu3(csrMatrix, csrMatrix, csrMatrix_d, csrMatrix_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 3)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        if(runGPUVersion4) {

            printf("Running GPU version 4\n");

            // Reset
            clearCOOMatrixOnGPU(cooMatrix_d);
            cudaDeviceSynchronize();

            // Compute on GPU with version 4
            startTime(&timer);
            spmspm_gpu4(csrMatrix, csrMatrix, csrMatrix_d, csrMatrix_d, cooMatrix_d);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    GPU kernel time (version 4)", GREEN);

            // Copy data from GPU
            startTime(&timer);
            copyCOOMatrixFromGPU(cooMatrix_d, cooMatrix_h);
            cudaDeviceSynchronize();
            stopTime(&timer);
            printElapsedTime(timer, "    Copy from GPU time");

            // Verify
            verify(cooMatrix_h, cooMatrix, quickVerify);

        }

        // Free GPU memory
        startTime(&timer);
        freeCSRMatrixOnGPU(csrMatrix_d);
        freeCOOMatrixOnGPU(cooMatrix_d);
        cudaDeviceSynchronize();
        stopTime(&timer);
        printElapsedTime(timer, "GPU deallocation time");

    }

    // Free memory
    freeCSRMatrix(csrMatrix);
    freeCOOMatrix(cooMatrix);

    return 0;

}

