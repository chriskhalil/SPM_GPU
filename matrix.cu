
#include "matrix.h"

#include <assert.h>
#include <cstdlib>
#include <stdio.h>

COOMatrix* createEmptyCOOMatrix(unsigned int numRows, unsigned int numCols, unsigned int capacity) {
    COOMatrix *cooMatrix = (COOMatrix *) malloc(sizeof(COOMatrix));
    cooMatrix->numRows = numRows;
    cooMatrix->numCols = numCols;
    cooMatrix->numNonzeros = 0;
    cooMatrix->capacity = capacity;
    cooMatrix->rowIdxs = (unsigned int*) calloc(1, capacity*sizeof(unsigned int));
    cooMatrix->colIdxs = (unsigned int*) malloc(capacity*sizeof(unsigned int));
    cooMatrix->values = (float*) malloc(capacity*sizeof(float));
    return cooMatrix;
}

void freeCOOMatrix(COOMatrix* cooMatrix) {
    free(cooMatrix->rowIdxs);
    free(cooMatrix->colIdxs);
    free(cooMatrix->values);
    free(cooMatrix);
}

void merge(unsigned int* key1, unsigned int *key2, float *data, int left, int mid, int right, unsigned int* key1Aux, unsigned int* key2Aux, float* dataAux) {

    int size1 = mid - left + 1;
    int size2 = right - mid;

    unsigned int *key1Left  = key1Aux; key1Aux += size1;
    unsigned int *key1Right = key1Aux; key1Aux += size2;
    unsigned int *key2Left  = key2Aux; key2Aux += size1;
    unsigned int *key2Right = key2Aux; key2Aux += size2;
    float *dataLeft  = dataAux; dataAux += size1;
    float *dataRight = dataAux; dataAux += size2;

    for(int i = 0; i < size1; ++i) {
        key1Left[i] = key1[left + i];
        key2Left[i] = key2[left + i];
        dataLeft[i] = data[left + i];
    }
    for(int j = 0; j < size2; ++j) {
        key1Right[j] = key1[mid + 1 + j];
        key2Right[j] = key2[mid + 1 + j];
        dataRight[j] = data[mid + 1 + j];
    }

    int i = 0, j = 0, k = left;
    while (i < size1 && j < size2) {
        if (key1Left[i] < key1Right[j] || key1Left[i] == key1Right[j] && key2Left[i] <= key2Right[j]) {
            key1[k] = key1Left[i];
            key2[k] = key2Left[i];
            data[k] = dataLeft[i];
            ++i;
            ++k;
        } else {
            key1[k] = key1Right[j];
            key2[k] = key2Right[j];
            data[k] = dataRight[j];
            ++j;
            ++k;
        }
    }
    while (i < size1) {
        key1[k] = key1Left[i];
        key2[k] = key2Left[i];
        data[k] = dataLeft[i];
        ++i;
        ++k;
    }
    while (j < size2) {
        key1[k] = key1Right[j];
        key2[k] = key2Right[j];
        data[k] = dataRight[j];
        ++j;
        ++k;
    }

}

void mergeSort(unsigned int *key1, unsigned int *key2, float *data, int start, int end, unsigned int* key1Aux, unsigned int* key2Aux, float* dataAux) {
    if (start < end) {
        int mid = start + (end - start) / 2;
        mergeSort(key1, key2, data, start, mid, key1Aux, key2Aux, dataAux);
        mergeSort(key1, key2, data, mid + 1, end, key1Aux, key2Aux, dataAux);
        merge(key1, key2, data, start, mid, end, key1Aux, key2Aux, dataAux);
    }
}

void sortCOOMatrix(COOMatrix* cooMatrix) {
    unsigned int* key1Aux = (unsigned int*) malloc(cooMatrix->numNonzeros*sizeof(unsigned int));
    unsigned int* key2Aux = (unsigned int*) malloc(cooMatrix->numNonzeros*sizeof(unsigned int));
    float* dataAux = (float*) malloc(cooMatrix->numNonzeros*sizeof(float));
    mergeSort(cooMatrix->rowIdxs, cooMatrix->colIdxs, cooMatrix->values, 0, cooMatrix->numNonzeros - 1, key1Aux, key2Aux, dataAux);
    free(key1Aux);
    free(key2Aux);
    free(dataAux);
}

COOMatrix* createEmptyCOOMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int capacity) {

    COOMatrix cooMatrixShadow;
    cooMatrixShadow.numRows = numRows;
    cooMatrixShadow.numCols = numCols;
    cooMatrixShadow.numNonzeros = 0;
    cooMatrixShadow.capacity = capacity;
    cudaMalloc((void**) &cooMatrixShadow.rowIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooMatrixShadow.colIdxs, capacity*sizeof(unsigned int));
    cudaMalloc((void**) &cooMatrixShadow.values, capacity*sizeof(float));

    COOMatrix* cooMatrix;
    cudaMalloc((void**) &cooMatrix, sizeof(COOMatrix));
    cudaMemcpy(cooMatrix, &cooMatrixShadow, sizeof(COOMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    return cooMatrix;

}

void clearCOOMatrixOnGPU(COOMatrix* cooMatrix) {
    COOMatrix cooMatrixShadow;
    cudaMemcpy(&cooMatrixShadow, cooMatrix, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    cudaMemset(cooMatrixShadow.rowIdxs, 0, cooMatrixShadow.numNonzeros*sizeof(unsigned int));
    cudaMemset(cooMatrixShadow.colIdxs, 0, cooMatrixShadow.numNonzeros*sizeof(unsigned int));
    cudaMemset(cooMatrixShadow.values, 0, cooMatrixShadow.numNonzeros*sizeof(unsigned int));
    cudaMemset(&cooMatrix->numNonzeros, 0, sizeof(unsigned int));
}

void freeCOOMatrixOnGPU(COOMatrix* cooMatrix) {
    COOMatrix cooMatrixShadow;
    cudaMemcpy(&cooMatrixShadow, cooMatrix, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    cudaFree(cooMatrixShadow.rowIdxs);
    cudaFree(cooMatrixShadow.colIdxs);
    cudaFree(cooMatrixShadow.values);
    cudaFree(cooMatrix);
}

void copyCOOMatrixFromGPU(COOMatrix* cooMatrix_d, COOMatrix* cooMatrix_h) {
    COOMatrix cooMatrixShadow;
    cudaMemcpy(&cooMatrixShadow, cooMatrix_d, sizeof(COOMatrix), cudaMemcpyDeviceToHost);
    assert(cooMatrix_h->numRows == cooMatrixShadow.numRows);
    assert(cooMatrix_h->numCols == cooMatrixShadow.numCols);
    assert(cooMatrix_h->capacity >= cooMatrixShadow.numNonzeros);
    cooMatrix_h->numNonzeros = cooMatrixShadow.numNonzeros;
    cudaMemcpy(cooMatrix_h->rowIdxs, cooMatrixShadow.rowIdxs, cooMatrixShadow.numNonzeros*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooMatrix_h->colIdxs, cooMatrixShadow.colIdxs, cooMatrixShadow.numNonzeros*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cooMatrix_h->values, cooMatrixShadow.values, cooMatrixShadow.numNonzeros*sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
}

COOMatrix* readCOOMatrix(const char* fileName) {

    COOMatrix* cooMatrix = (COOMatrix*) malloc(sizeof(COOMatrix));

    // Initialize fields
    FILE* fp = fopen(fileName, "r");
    int x = 1;
    x |= fscanf(fp, "%u", &cooMatrix->numRows);
    cooMatrix->numCols = cooMatrix->numRows;
    x |= fscanf(fp, "%u", &cooMatrix->numNonzeros);
    cooMatrix->rowIdxs = (unsigned int*) malloc(cooMatrix->numNonzeros*sizeof(unsigned int));
    cooMatrix->colIdxs = (unsigned int*) malloc(cooMatrix->numNonzeros*sizeof(unsigned int));
    cooMatrix->values = (float*) malloc(cooMatrix->numNonzeros*sizeof(float));

    // Read the nonzeros
    for(unsigned int i = 0; i < cooMatrix->numNonzeros; ++i) {
        x |= fscanf(fp, "%u", &cooMatrix->rowIdxs[i]);
        x |= fscanf(fp, "%u", &cooMatrix->colIdxs[i]);
        cooMatrix->values[i] = 0.5 + ((float)rand())/RAND_MAX;
    }

    return cooMatrix;

}

CSRMatrix* coo2csr(COOMatrix* cooMatrix) {

    CSRMatrix* csrMatrix = (CSRMatrix*) malloc(sizeof(CSRMatrix));;

    // Initialize fields
    csrMatrix->numRows = cooMatrix->numRows;
    csrMatrix->numCols = cooMatrix->numCols;
    csrMatrix->numNonzeros = cooMatrix->numNonzeros;
    csrMatrix->rowPtrs = (unsigned int*) malloc((csrMatrix->numRows + 1)*sizeof(unsigned int));
    csrMatrix->colIdxs = (unsigned int*) malloc(csrMatrix->numNonzeros*sizeof(unsigned int));
    csrMatrix->values = (float*) malloc(csrMatrix->numNonzeros*sizeof(float));

    // Histogram rows
    memset(csrMatrix->rowPtrs, 0, (csrMatrix->numRows + 1)*sizeof(unsigned int));
    for(unsigned int i = 0; i < cooMatrix->numNonzeros; ++i) {
        unsigned int row = cooMatrix->rowIdxs[i];
        csrMatrix->rowPtrs[row]++;
    }

    // Prefix sum rowPtrs
    unsigned int sumBeforeNextRow = 0;
    for(unsigned int row = 0; row < csrMatrix->numRows; ++row) {
        unsigned int sumBeforeRow = sumBeforeNextRow;
        sumBeforeNextRow += csrMatrix->rowPtrs[row];
        csrMatrix->rowPtrs[row] = sumBeforeRow;
    }
    csrMatrix->rowPtrs[csrMatrix->numRows] = sumBeforeNextRow;

    // Bin the nonzeros
    for(unsigned int i = 0; i < cooMatrix->numNonzeros; ++i) {
        unsigned int row = cooMatrix->rowIdxs[i];
        unsigned int j = csrMatrix->rowPtrs[row]++;
        csrMatrix->colIdxs[j] = cooMatrix->colIdxs[i];
        csrMatrix->values[j] = cooMatrix->values[i];
    }

    // Restore rowPtrs
    for(unsigned int row = csrMatrix->numRows - 1; row > 0; --row) {
        csrMatrix->rowPtrs[row] = csrMatrix->rowPtrs[row - 1];
    }
    csrMatrix->rowPtrs[0] = 0;

    return csrMatrix;

}

CSRMatrix* createCSRMatrixFromFile(const char* fileName) {
    COOMatrix* cooMatrix = readCOOMatrix(fileName);
    CSRMatrix* csrMatrix = coo2csr(cooMatrix);
    freeCOOMatrix(cooMatrix);
    return csrMatrix;
}

void freeCSRMatrix(CSRMatrix* csrMatrix) {
    free(csrMatrix->rowPtrs);
    free(csrMatrix->colIdxs);
    free(csrMatrix->values);
    free(csrMatrix);
}

CSRMatrix* createEmptyCSRMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int numNonzeros) {

    CSRMatrix csrMatrixShadow;
    csrMatrixShadow.numRows = numRows;
    csrMatrixShadow.numCols = numCols;
    csrMatrixShadow.numNonzeros = numNonzeros;
    cudaMalloc((void**) &csrMatrixShadow.rowPtrs, (numRows + 1)*sizeof(unsigned int));
    cudaMalloc((void**) &csrMatrixShadow.colIdxs, numNonzeros*sizeof(unsigned int));
    cudaMalloc((void**) &csrMatrixShadow.values, numNonzeros*sizeof(float));

    CSRMatrix* csrMatrix;
    cudaMalloc((void**) &csrMatrix, sizeof(CSRMatrix));
    cudaMemcpy(csrMatrix, &csrMatrixShadow, sizeof(CSRMatrix), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    return csrMatrix;

}

void freeCSRMatrixOnGPU(CSRMatrix* csrMatrix) {
    CSRMatrix csrMatrixShadow;
    cudaMemcpy(&csrMatrixShadow, csrMatrix, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    cudaFree(csrMatrixShadow.rowPtrs);
    cudaFree(csrMatrixShadow.colIdxs);
    cudaFree(csrMatrixShadow.values);
    cudaFree(csrMatrix);
}

void copyCSRMatrixToGPU(CSRMatrix* csrMatrix_h, CSRMatrix* csrMatrix_d) {
    CSRMatrix csrMatrixShadow;
    cudaMemcpy(&csrMatrixShadow, csrMatrix_d, sizeof(CSRMatrix), cudaMemcpyDeviceToHost);
    assert(csrMatrixShadow.numRows == csrMatrix_h->numRows);
    assert(csrMatrixShadow.numCols == csrMatrix_h->numCols);
    assert(csrMatrixShadow.numNonzeros == csrMatrix_h->numNonzeros);
    cudaMemcpy(csrMatrixShadow.rowPtrs, csrMatrix_h->rowPtrs, (csrMatrix_h->numRows + 1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrMatrixShadow.colIdxs, csrMatrix_h->colIdxs, csrMatrix_h->numNonzeros*sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(csrMatrixShadow.values, csrMatrix_h->values, csrMatrix_h->numNonzeros*sizeof(float), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
}

