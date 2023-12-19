#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

FILE* openOutputFile(int argc, char* const* argv, int totalProcs);
void initializeArrays(double* matrix, double* vector, double* result, int rows, int cols);
void matrixVectorMultiplication(double* matrix, double* vector, double* result, int numCols, int startRow, int endRow, int startCol, int endCol);
void resetVector(double* vector, int length);
void reportElapsedTime(int processId, double elapsedTime);

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
        printf("Number of processes: %d\nMatrix: %dx%d\n", processCount, matrixRows, matrixCols);
    }

    int vecSize = matrixCols;
    int blocks1d = (int)sqrt(processCount);
    int totalBlocks = blocks1d * blocks1d;
    int blockRows = matrixRows / blocks1d;
    int blockCols = matrixCols / blocks1d;

    double* matrix = (double*)malloc(matrixRows * matrixCols * sizeof(double));
    double* vector = (double*)malloc(vecSize * sizeof(double));
    double* partialResult = (double*)malloc(matrixRows * sizeof(double));
    double* finalResult = NULL;

    if (processId == 0) {
        finalResult = (double*)malloc(matrixRows * sizeof(double));
        initializeArrays(matrix, vector, finalResult, matrixRows, matrixCols);
    }

    MPI_Bcast(vector, matrixCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, matrixRows * matrixCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int startRow = (processId / blocks1d) * blockRows;
    int endRow = startRow + blockRows;
    int startCol = (processId % blocks1d) * blockCols;
    int endCol = startCol + blockCols;

    if (processId < totalBlocks) {
        if (processId == totalBlocks - 1) {
            endRow = matrixRows;
            endCol = matrixCols;
        } else if ((processId + 1) % blocks1d == 0) {
            endCol = matrixCols;
        } else if ((processId + 1) > totalBlocks - blocks1d) {
            endRow = matrixRows;
        }
    }

    // Test performance
    int iterations = 100;
    for (size_t i = 0; i < iterations; ++i) {
        beginTime = MPI_Wtime();
        if (processId < totalBlocks) {
            matrixVectorMultiplication(matrix, vector, partialResult, matrixCols, startRow, endRow, startCol, endCol);
        }
        endTime = MPI_Wtime();
        totalTime += endTime - beginTime;
        resetVector(partialResult, matrixRows);
    }
    totalTime /= iterations;

    // Actual run
    if (processId < totalBlocks) {
        matrixVectorMultiplication(matrix, vector, partialResult, matrixCols, startRow, endRow, startCol, endCol);
    }

    // Aggregate the results of all processes
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(partialResult, finalResult, matrixRows, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (processId == 0) {
        if (outputFile != NULL) {
            fprintf(outputFile, "%f ", totalTime);
        }
        printf("\n");
    }

    reportElapsedTime(processId, totalTime);

    fclose(outputFile);

    free(matrix);
    free(vector);
    free(partialResult);
    if (processId == 0) {
        free(finalResult);
    }

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

void matrixVectorMultiplication(double* matrix, double* vector, double* result, int numCols, int startRow, int endRow, int startCol, int endCol) {
    for (size_t i = startRow; i < endRow; ++i) {
        for (size_t j = startCol; j < endCol; ++j) {
            result[i] += matrix[i * numCols + j] * vector[j];
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