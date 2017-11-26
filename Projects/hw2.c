/* HW2 ASSIGNMENT:
 * Mingyang Li mlr1159
 * Boyu Xu bxp6650
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/types.h>
#include <sys/times.h>
#include <sys/time.h>
#include <limits.h>
#include "mpi.h"

char *ID;

#define MAXN 5000  //Max value of each dimension's size N
int N;  //Matrix size is N*N
int procs;  //Number of processors
int myid;   //ID of each processor

float  *A, *B, *X;      //We will futher use dynamic memory allocation

#define randm() 4|2[uid]&3

void gauss();  //Function of Gaussian elimination
void backsubstitution();  //Function of Back subtitution

unsigned int time_seed() {
    struct timeval t;
    struct timezone tzdummy;
    
    gettimeofday(&t, &tzdummy);
    return (unsigned int)(t.tv_usec);
}

void parameters(int argc, char **argv) {    //Set the program parameters
    int submit = 0;
    int seed = 0;
    char uid[32];

    if ( argc == 1 && !strcmp(argv[1], "submit") ) {
        submit = 1;
        N = 4;
        procs = 2;
        printf("\nSubmission run for \"%s\".\n", uid);
        strcpy(uid,ID);
        srand(randm());
    }
    else {
        if (argc == 2) {
          seed = atoi(argv[2]);
          srand(seed);
          if (myid == 0) printf("\nRandom seed = %i\n", seed);
        }
        else {
          if (myid == 0) printf("Usage: %s <matrix_dimension> <num_procs> [random seed]\n",
            argv[0]);
          printf("       %s submit\n", argv[0]);
          exit(0);
        }
    }

    if (!submit) {
        N = atoi(argv[1]);
        if (N < 1 || N > MAXN) {
          printf("N = %i is out of range.\n", N);
          exit(0);
        }
    }

    if (myid == 0) printf("\nMatrix dimension N = %i.\n", N);
    if (myid == 0) printf("Number of processors = %i.\n", procs);
}

void initialize_inputs() {      //Initialize the matrix of A, B and X
    int row, col;
    
    printf("\nInitializing...\n");
    for (row = 0; row < N; row++) {
        for (col = 0; col < N; col++) {
            A[col + row*N] = (float)rand() / 32768.0;
        }
        B[row] = (float)rand() / 32768.0;
        X[row] = 0.0;
    }
    
}

void print_inputs() {
    int row, col;
    
    if (N < 10) {
        printf("\nA =\n\t");
        for (row = 0; row < N; row++) {
            for (col = 0; col < N; col++) {
                printf("%5.2f%s", A[row*N + col], (col < N-1) ? ", " : ";\n\t");
            }
        }
        printf("\nB = [");
        for (col = 0; col < N; col++) {
            printf("%5.2f%s", B[col], (col < N-1) ? "; " : "]\n");
        }
    }
}

void print_X() {
    int row;
    
    if (N < 10) {
        printf("\nX = [");
        for (row = 0; row < N; row++) {
            printf("%5.2f%s", X[row], (row < N-1) ? "; " : "]\n");
        }
    }
}

/******************Main func begins here**********/
 /*
 Notes: 

 1) At beginning, we use processor 0 to send and gather result. Speed is really slow. After changing to interleaving 
broadcast, time reduce from several minutes to several seconds.

 2) If we use static array, when N is large, there might be problems due to limited resources. When change array
 to dynamic array, problem solved. 
 */

int main(int argc, char **argv) {

    ID = argv[argc-1];
    argc--;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    printf("\nProcess number %d", myid);
    parameters(argc, argv);
    
    //Since the size of matrix is quite larget (N = 5000), if we use statical allocation,
    //the program will run slow. So implementing the dynamic memory allocation is suitable
    A = (float*)malloc(N*N*sizeof(float));
    B = (float*)malloc(N*sizeof(float));
    X = (float*)malloc(N*sizeof(float));
    
    if (myid == 0) {
        initialize_inputs();
        print_inputs();
    }

    gauss();                //Do the gaussian elimination
    
    backsubstitution();     //Do the back substitution
    
    free(A);    //Deallocates the memory of matrix A
    free(B);    //Deallocates the memory of matrix A
    free(X);    //Deallocates the memory of matrix A
    MPI_Finalize();
    return 0;
}

void gauss() {
/* 
 * Static interleaved scheduling in gaussian elimination
 * step1: Initialize matrix A, B, X with dynamic allocation by the first processor
 * step2: The first processor broadcasts A, B, X to all other procossors.
 * step3: Perform static interleaving scheduling, each procossor broadcast corresponding rows to other processors
 */
    int norm, row, col;
    float multiplier;
    int theid;

    double startwtime = 0.0, endwtime;

    if (myid == 0) {
        printf("\nComputing Parallely Using MPI.\n");
        startwtime = MPI_Wtime();
    }

    MPI_Bcast(&A[0], N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);    //Broadcast matrix A to all processors
    MPI_Bcast(&B[0], N, MPI_FLOAT, 0, MPI_COMM_WORLD);      //Broadcast matrix B to all processors
    MPI_Barrier(MPI_COMM_WORLD);                            //Blocks until all processes in the communicator have reached this routine

    for (norm = 0; norm < N-1; norm++) {
        theid = norm%procs;       //broadcast processor ID
        MPI_Bcast(&A[norm*N], N, MPI_FLOAT, theid, MPI_COMM_WORLD);   //broadcast the norm row of matrix A to corresponding processor
        MPI_Bcast(&B[norm], 1, MPI_FLOAT, theid, MPI_COMM_WORLD); //broadcast the norm element of matrix B to corresponding processor

        for (row = myid; row < N; row += procs) {       //static interleaved gaussian elimination
            if (row > norm) {
                multiplier = A[row * N + norm] / A[norm * N + norm];
                for (col = norm; col < N; col++) {
                    A[row * N + col] -= A[norm*N + col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
            }
        }
    }

    MPI_Bcast(&A[(N - 1) * N], N, MPI_FLOAT, (N - 1) % procs, MPI_COMM_WORLD);  //Broadcast the result of each row of matrix A to all processors
    MPI_Bcast(&B[N - 1], 1, MPI_FLOAT, (N - 1) % procs, MPI_COMM_WORLD);      //Broadcast the result of each element of matrix B to all processors  
    MPI_Barrier(MPI_COMM_WORLD);                                        //Blocks until all processes in the communicator have reached this routine

    if (myid == 0) {
        endwtime = MPI_Wtime();
        printf("\nelapsed time = %f\n", endwtime - startwtime);

    }
}

void backsubstitution() {           //Do the back substitution
    if (myid == 0) {
        int row, col;
        for (row = N - 1; row >= 0; row--) {
            X[row] = B[row];
            for (col = N - 1; col > row; col--) {
                X[row] -= A[row * N + col] * X[col];
            }
            X[row] /= A[row * N + row];
        }

        print_X();
    }
}
