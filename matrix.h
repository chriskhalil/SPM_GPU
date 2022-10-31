
#ifndef __MATRIX_H_
#define __MATRIX_H_

struct COOMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int capacity;
    unsigned int* rowIdxs;
    unsigned int* colIdxs;
    float* values;
};

COOMatrix* createEmptyCOOMatrix(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void freeCOOMatrix(COOMatrix* cooMatrix);

void sortCOOMatrix(COOMatrix* cooMatrix);

COOMatrix* createEmptyCOOMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int capacity);
void freeCOOMatrixOnGPU(COOMatrix* cooMatrix);

void clearCOOMatrixOnGPU(COOMatrix* cooMatrix);
void copyCOOMatrixFromGPU(COOMatrix* cooMatrix_d, COOMatrix* cooMatrix_h);

struct CSRMatrix {
    unsigned int numRows;
    unsigned int numCols;
    unsigned int numNonzeros;
    unsigned int* rowPtrs;
    unsigned int* colIdxs;
    float* values;
};

CSRMatrix* createCSRMatrixFromFile(const char* fileName);
void freeCSRMatrix(CSRMatrix* csrMatrix);

CSRMatrix* createEmptyCSRMatrixOnGPU(unsigned int numRows, unsigned int numCols, unsigned int numNonzeros);
void freeCSRMatrixOnGPU(CSRMatrix* csrMatrix);

void copyCSRMatrixToGPU(CSRMatrix* csrMatrix_h, CSRMatrix* csrMatrix_d);

#endif

