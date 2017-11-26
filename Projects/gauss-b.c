/* ------------------------------- */

/*
 * In this method, we use #pragma pallel for a shared memory
 * parallel algorithm.
 */

void gauss() {
    int norm, row, col;
    float multiplier;

    for (norm = 0; norm < N - 1; norm++) {
        #pragma omp parallel private(row, col, multiplier) num_threads(procs)
        {
            #pragma omp for schedule(dynamic)
            for (row = norm + 1; row < N; row++) {
                multiplier = A[row][norm] / A[norm][norm];
                for (col = norm; col < N; col++) {
                    A[row][col] -= A[norm][col] * multiplier;
                }
                B[row] -= B[row] * multiplier;
            }
        }
    }

    printf("Computing Serially.\n");

    for (row = N - 1; row >= 0; row--) {
        X[row] = B[row];
        for (col = N - 1; col > row; col--) {
            X[row] -= A[row][col] * X[col];
        }
        X[row] /= A[row][row];
    }
