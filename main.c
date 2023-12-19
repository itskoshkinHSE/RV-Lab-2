#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include </usr/local/Cellar/open-mpi/5.0.0/include/mpi.h> //CHANGE_ME

void printMatrix(double *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%.1f ", matrix[i * size + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void fillArrays(int N, double *A, double *B, double *C)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            A[i * N + j] = 5;
            B[i * N + j] = 6;
            C[i * N + j] = 0.0;
        }
    }
}

void multiplyMatrixBlock(double *A, double *B, double *C, int N)
{
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                C[i * N + j] += A[i * N + k] * B[k * N + j];
}

int main(int argc, char *argv[])
{
    MPI_Comm cannonComm;
    int rank, size;
    int dims[2];
    int periods[2];
    int left, right, up, down;
    double *A, *B, *C;
    double *buf, *tmp;
    double start, end;
    unsigned int iseed = 0;
    int N = 2500;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Matrix size: %d\n", N);
        printf("Number of processes: %d\n", size);
    }
    srand(iseed);
    dims[0] = 0;
    dims[1] = 0;
    periods[0] = 1;
    periods[1] = 1;
    MPI_Dims_create(size, 2, dims);
    if (dims[0] != dims[1])
    {
        if (rank == 0)
        {
            printf("The number of processors must be a square.\n");
        }
        MPI_Finalize();
        return 0;
    }
    // size of block
    int localN = N / dims[0];

    A = (double *)malloc(localN * localN * sizeof(double));
    B = (double *)malloc(localN * localN * sizeof(double));
    buf = (double *)malloc(localN * localN * sizeof(double));
    C = (double *)calloc(localN * localN, sizeof(double));
    fillArrays(localN, A, B, C);

    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cannonComm);
    MPI_Cart_shift(cannonComm, 0, 1, &left, &right);
    MPI_Cart_shift(cannonComm, 1, 1, &up, &down);

    start = MPI_Wtime();
    for (int shift = 0; shift < dims[0]; shift++)
    {
        multiplyMatrixBlock(A, B, C, localN);

        if (shift == dims[0] - 1)
            break;

        // Communication
        MPI_Status status;
        MPI_Sendrecv(A, localN * localN, MPI_DOUBLE, left, 1, buf, localN * localN, MPI_DOUBLE, right, 1, cannonComm, &status);
        tmp = buf;
        buf = A;
        A = tmp;
        MPI_Sendrecv(B, localN * localN, MPI_DOUBLE, up, 2, buf, localN * localN, MPI_DOUBLE, down, 2, cannonComm, &status);
        tmp = buf;
        buf = B;
        B = tmp;
    }
    MPI_Barrier(cannonComm);
    end = MPI_Wtime();
    if (rank == 0)
    {
        printf("Time: %.4fs\n", end - start);
    }
    // Free memory
    free(A);
    free(B);
    free(buf);
    free(C);
    MPI_Finalize();
    return 0;
}