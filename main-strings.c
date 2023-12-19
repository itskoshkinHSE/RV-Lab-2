#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define EPSILON 0.000001

FILE* openOutputFile(int argc, char* const* argv, int totalProcs);
void initializeArrays(double* matrix, double* vector, double* result, int rows, int cols);
void matrixVectorMultiplication(double* localMatrix, double* localVector, double* localResult, int localRows, int cols);
void resetVector(double* vector, int length);
void reportElapsedTime(int processId, double elapsedTime);
void verifyResults(double* matrix, double* vector, double* result, int rows, int cols);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int processCount, processId;
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);

    FILE* outputFile = openOutputFile(argc, argv, processCount);

    double beginTime, endTime, totalTime = 0;
    int matrixRows = 5000, matrixCols = 5000;

    if (argc > 2) {
        matrixRows = atoi(argv[1]);
        matrixCols = atoi(argv[2]);
    } else if (processId == 0) {
        printf("WARNING: Matrix sizes are not provided. Using default 5000x5000\n");
    }

    if (processId == 0) {
        printf("Number of processes: %d\n", processCount);
        printf("Matrix: %dx%d\n", matrixRows, matrixCols);
    }

    int vecSize = matrixCols;
    int localRows = matrixRows / processCount;
    int localElements = localRows * matrixCols;

    double *matrix = NULL, *vector = NULL, *localMatrix = NULL, *localResult = NULL, *globalResult = NULL;
    vector = (double*)malloc(vecSize * sizeof(double));
    localMatrix = (double*)malloc(localElements * sizeof(double));
    localResult = (double*)malloc(localRows * sizeof(double));
    if (processId == 0) {
        matrix = (double*)malloc(matrixRows * matrixCols * sizeof(double));
        globalResult = (double*)malloc(matrixRows * sizeof(double));
        initializeArrays(matrix, vector, globalResult, matrixRows, matrixCols);
    }

    MPI_Scatter(matrix, localElements, MPI_DOUBLE, localMatrix, localElements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, vecSize, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int iters = 100;
    for (int i = 0; i < iters; ++i) {
        beginTime = MPI_Wtime();
        matrixVectorMultiplication(localMatrix, vector, localResult, localRows, matrixCols);
        endTime = MPI_Wtime();
        totalTime += endTime - beginTime;
        resetVector(localResult, localRows);
    }
    totalTime /= (double)iters;

    matrixVectorMultiplication(localMatrix, vector, localResult, localRows, matrixCols);

    MPI_Gather(localResult, localRows, MPI_DOUBLE, globalResult, localRows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (processId == 0) {
        int remainRows = matrixRows % processCount;
        int offset = matrixRows - remainRows;
        beginTime = MPI_Wtime();
        matrixVectorMultiplication(matrix + matrixCols * offset, vector, globalResult + offset, remainRows, matrixCols);
        endTime = MPI_Wtime();
        totalTime += endTime - beginTime;
    }

    if (outputFile) {
        if (processId == 0) {
            verifyResults(matrix, vector, globalResult, matrixRows, matrixCols);
            if (outputFile != NULL) {
                fprintf(outputFile, "%f ", totalTime);
            }
            printf("\n");
        }
    }

    reportElapsedTime(processId, totalTime);

    fclose(outputFile);

    free(matrix);
    free(globalResult);
    free(vector);
    free(localMatrix);
    free(localResult);

    MPI_Finalize();
    return 0;
}

FILE* openOutputFile(int argc, char* const* argv, int totalProcs) {
    FILE* fileStream = NULL;
    if (argc == 4) {
        fileStream = fopen(argv[3], totalProcs == 1 ? "w" : "a");
    }
    return fileStream;
}

void initializeArrays(double* matrix, double* vector, double* result, int rows, int cols) {
    srand(time(NULL));

    for (int i = 0; i < rows * cols; ++i) {
        matrix[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    for (int i = 0; i < cols; ++i) {
        vector[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    }
    for (int i = 0; i < rows; ++i) {
        result[i] = 0.0;
    }
}

void matrixVectorMultiplication(double* localMatrix, double* localVector, double* localResult, int localRows, int cols) {
    for (size_t i = 0; i < localRows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            localResult[i] += localMatrix[i * cols + j] * localVector[j];
        }
    }
}

void resetVector(double* vector, int length) {
    for (int i = 0; i < length; ++i) {
        vector[i] = 0;
    }
}

void reportElapsedTime(int processId, double elapsedTime) {
    double maxElapsedTime;
    MPI_Reduce(&elapsedTime, &maxElapsedTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (processId == 0) {
        printf("Maximum processing time across all processes: %.3f ms\n", maxElapsedTime * 1000.0);
    }
}

void verifyResults(double* matrix, double* vector, double* result, int rows, int cols) {
    double* referenceResult = (double*)malloc(rows * sizeof(double));
    resetVector(referenceResult, rows);

    matrixVectorMultiplication(matrix, vector, result, rows, cols);

    for (int i = 0; i < rows; ++i) {
        if (fabs(result[i] - referenceResult[i]) > EPSILON) {
            printf("Discrepancy found at row %d: calculated = %f, expected = %f\n", i, result[i], referenceResult[i]);
        }
    }

    free(referenceResult);
}