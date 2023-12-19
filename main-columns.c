#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

#define EPSILON 0.000001

FILE* openOutputFile(int argc, char* const* argv, int totalProcs);
void initializeArrays(double* matrix, double* vector, double* result, int rows, int cols);
void matrixVectorMultiplication(double* matrix, int matrixCols, int matrixRows, int startColumn, int endColumn, double *vector, double* result);
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
        printf("Error: Please provide matrix dimensions. Defaulting to 5000x5000.\n");
    }

    if (processId == 0) {
        printf("Number of processes: %d\n", processCount);
        printf("Matrix dimensions: %dx%d\n", matrixRows, matrixCols);
    }

    double* matrix = (double*)malloc(matrixRows * matrixCols * sizeof(double));
    double* vector = (double*)malloc(matrixCols * sizeof(double));
    double* partialResult = (double*)malloc(matrixRows * sizeof(double));
    double* finalResult = NULL;

    resetVector(partialResult, matrixRows);

    if (processId == 0) {
        finalResult = (double*)malloc(matrixRows * sizeof(double));
        initializeArrays(matrix, vector, finalResult, matrixRows, matrixCols);
    }

    MPI_Bcast(vector, matrixCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(matrix, matrixRows * matrixCols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int columnsPerProcess = matrixCols / processCount;
    int startColumn = processId * columnsPerProcess;
    int endColumn = (processId == processCount - 1) ? matrixCols : startColumn + columnsPerProcess;

    int iterations = 100;
    for (int i = 0; i < iterations; ++i) {
        beginTime = MPI_Wtime();
        matrixVectorMultiplication(matrix, matrixCols, matrixRows, startColumn, endColumn, vector, partialResult);
        endTime = MPI_Wtime();
        totalTime += endTime - beginTime;
        resetVector(partialResult, matrixRows);
    }
    totalTime /= iterations;

    if (processId < matrixCols) {
        matrixVectorMultiplication(matrix, matrixCols, matrixRows, startColumn, endColumn, vector, partialResult);
    }

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

void matrixVectorMultiplication(double* matrix, int matrixCols, int matrixRows, int startColumn, int endColumn, double *vector, double* result) {
    for (int row = 0; row < matrixRows; ++row) {
        for (int col = startColumn; col < endColumn; ++col) {
            result[row] += matrix[row * matrixCols + col] * vector[col];
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