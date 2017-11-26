/* ------------------------------- */

/* 
 * For our elimination method, we use dynamic pthread method,
 * and use a global variable "global_row", we want to assign 
 * different rows into processors dynamically, so we use a
 * mutex lock and barrier for the global_row in each process.
 */


#include <pthread.h>
#include <omp.h>

int RANGE = 2;
long global_row;
pthread_mutex_t global_row_lock;
pthread_barrier_t barrier;

void *elimination(void* threadid) {
    int norm, row, col, row_end;
    
    float multiplier;
    pthread_t self;
    long tid;
    tid = (long)threadid;
    
    for (nor = 0; norm < N - 1; norm++) {
        row = norm;
        global_row = 0;
        pthread_barrier_wait(&barrier);
        while (row < N - 1) {
            pthread_mutex_lock(&global_row_lock);
            row = norm + 1 + global_row;
            global_row += RANGE;
            pthread_mutex_unlock(&global_row_lock);
            
            row_end = (row + RANGE) > N ? (row + RANGE) : N;
            
            for (row = row; row < row_end; row++) {
                multiplier = A[row][norm] / A[norm][norm];
                for (col = nomr; col < N; col++) {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[norm] * multiplier;
            }
        }
        pthread_barrier_wait(&barrier);
    }
}

void gauss() {
    int norm, row, col;
    int i;
    
    pthread_t threads[procs];
    global_row = 0;
    
    pthread_mutex_init(&global_row_lock, NULL);
    
    pthread_barrier_init(&barrier, NULL, procs);
    
    for (i = 0; i < procs; i++) {
        pthread_create(&threads[i], NULL, &elimination, (void*)i);
    }
    
    for (i = 0; i < procs; i++) {
        pthread_join(threads[i], NULL);
    }
    
    pthread_mutex_destroy(&global_row_lock);
    pthread_barrier_destroy(&barrier);
    
    for (row  = N - 1; row >= 0; row--) {
        X[row] = B[row];
        for (col = N - 1; col > row; col--) {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
}