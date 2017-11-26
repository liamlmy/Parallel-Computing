#include "mpi.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NUM_POINTS 524288

unsigned int X_axis[NUM_POINTS];
unsigned int Y_axis[NUM_POINTS];

void swap(unsigned int array[], int i, int j);
unsigned int find_kth(unsigned int *v, int n, int k, unsigned int *y);
int i = 0;
int j = 0;
int k = 0;
int p = 0;
//global_cost computed at process 0
double global_cost = 0;

int numprocs;  /* Number of processors to use */
int myid;

int num_quadrants;
void find_quadrants (num_quadrants)
{  
   if (myid == 0) {
     //cut x
    int x_cut = 0;
    //current q
    int quadrants = 1;

    int top[num_quadrants];
    int left[num_quadrants];
    int right[num_quadrants];
    int bottom[num_quadrants];

    //bisection coordinates
    int pivot_arr[num_quadrants];
    while (num_quadrants > quadrants) { 
        int points_num = NUM_POINTS / quadrants;
        if (!x_cut) {
            for (i = 0; i < quadrants; i++) {
              //find median
                int x_pivot = find_kth(X_axis + i * points_num, points_num, points_num/2 - 1, Y_axis + i * points_num);
                k = i * points_num;
                j = i * points_num + points_num - 1;
                //small in the left of pivot
                 while (k < j && k < k + points_num / 2) {
                    if (x_pivot < X_axis[k]) {
                        while (X_axis[j] > x_pivot) {
                            j--;
                        }
                        if (k < j) {
                            swap(X_axis, k, j);
                            swap(Y_axis, k, j);
                        } 
                    }
                    k++;
                }
                pivot_arr[i+quadrants-1] = x_pivot;
            }
            x_cut = 1;  
        } else {
          //bisection on Y dimension 
            for (i = 0; i < quadrants; i++) {
                int y_pivot = find_kth(Y_axis + i * points_num, points_num, points_num/2 - 1, X_axis + i * points_num);

                k = i * points_num;
                j = i * points_num + points_num - 1;
                //make sure that the numbers smaller than th median are all gathered together on the left half of the array
                while (k < j &&  points_num >= 2) {
                    if (Y_axis[k] > y_pivot) {
                        while (Y_axis[j] > y_pivot) {
                            j--;
                        }
                        if (k < j) {
                            swap(X_axis, k, j);
                            swap(Y_axis, k, j);
                        } 
                    }
                    k++;
                }
                pivot_arr[i+quadrants-1] = y_pivot;
            }
            x_cut = 0;
        }
        quadrants *= 2;

    }

    //find border of initial quadrant by finding the border of X and Y
    int min_x = X_axis[0];
    int max_x = X_axis[0];
    int min_y = Y_axis[0];
    int max_y = Y_axis[0];
    for (i = 1; i < NUM_POINTS; i++) {
        if (X_axis[i] < min_x) {
            min_x = X_axis[i];
        }
        if (X_axis[i] > max_x) {
            max_x = X_axis[i];
        }
        if (Y_axis[i] < min_y) {
            min_y = Y_axis[i];
        }
        if (Y_axis[i] > max_y) {
            max_y = Y_axis[i];
        }
    }
    //update the quadrants' coordinates according to the values in the pivot_array[]
    top[0] = min_y;
    bottom[0] = max_y;
    left[0] = min_x;
    right[0] = max_x;
    i = 0;
    x_cut = 0;
    quadrants = 1;

    while (i < num_quadrants-1) {
        int temp = i;
        for (j = 0; j < quadrants; j++) {
            top[j+quadrants] = top[j];
            bottom[j+quadrants] = bottom[j];
            left[j+quadrants] = left[j];
            right[j+quadrants] = right[j];
        }
        if (!x_cut) {
            for (j = 0; j < quadrants * 2; j+=2) {
                top[j] = top[quadrants + j/2];
                bottom[j] = bottom[quadrants + j/2];
                left[j] = left[quadrants + j/2];
                right[j] = pivot_arr[temp];

                top[j+1] = top[quadrants + j/2];
                bottom[j+1] = bottom[quadrants + j/2];
                left[j+1] = pivot_arr[temp];
                right[j+1] = right[quadrants + j/2];
                temp++;
            }
            x_cut = 1;
        } else{
            for (j = 0; j < quadrants * 2; j+=2) {
                top[j] = top[quadrants + j/2];
                bottom[j] = pivot_arr[temp];
                left[j] = left[quadrants + j/2];
                right[j] = right[quadrants + j/2];
                
                top[j+1] = pivot_arr[temp];
                bottom[j+1] = bottom[quadrants + j/2];
                left[j+1] = left[quadrants + j/2];
                right[j+1] = right[quadrants + j/2];
                temp++;
            }
            x_cut = 0;
        }
        i += quadrants;
        quadrants *= 2;
    }
       printf("\nPrint quadrants coordinates: quarant number (top-left, top-right, bottom-left, bottom-righ)t\n");
        for (p = 0; p < num_quadrants; p++) {
            printf("\nNumber %d : " , p);
            printf(" (%d,%d) ", left[p], top[p]);
            printf(" (%d,%d) ", right[p], top[p]);
            printf(" (%d,%d) ", left[p], bottom[p]);
            printf(" (%d,%d) \n", right[p], bottom[p]);
        }
    }
    MPI_Bcast(&X_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD);  

    double local_cost = 0;
    for (i = myid; i < num_quadrants; i += numprocs)
    {
       int points = NUM_POINTS / num_quadrants;
       for (j = 0; j < points - 1; j++) {
            for (k = j+1; k < points; k++) {
                int x1 = points * i + j;
                int x2 = points * i + k;
                int y1 = points * i + j;
                int y2 = points * i + k;

                double diff_x = abs(X_axis[x1] - X_axis[x2]);
                double diff_y = abs(Y_axis[y1] - Y_axis[y2]);
                local_cost += sqrt((double)diff_x * diff_x + diff_y * diff_y);
            }
       }     
    }
    //reduce and calculate the global_cost on process 0
     MPI_Reduce(&local_cost, &global_cost, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

//find the kth smallest key in v
unsigned int find_kth(unsigned int *v, int n, int k, unsigned int *y) {
    int j0 = 0;
    int i1 = 0;
    int j1 = 0;
    int jmin = 0;
    if (n == 1 && k == 0) return v[0];
    int m = (n + 4)/5;
    unsigned int *medians =  (unsigned int *)malloc(m * sizeof(int));
    for (i1=0; i1<m; i1++) {
        if (5*i1 + 4 >= n){
            medians[i1] = v[5*i1];
        }
        else {
            unsigned int *w = v + 5*i1;
            unsigned int *y_s = y + 5*i1;
            for (j0=0; j0<3; j0++) {
                jmin = j0;
                for (j1=j0+1; j1<5; j1++) {
                    if (w[j1] < w[jmin]) {
                        jmin = j1;
                    }
                }
                swap(w, j0, jmin);
                swap(y_s, j0, jmin);
            }
            medians[i1] = w[2];
        } 
    }
    //find the median of medians
    int pivot = find_kth(medians, m, m/2, medians);
    free(medians);
 
    for (i1=0; i1<n; i1++) {
        if (v[i1] == pivot) {
            swap(v, i1, n-1);
            swap(y, i1, n-1);
            break;
        }
    }
    //point_s calculate point in pivot left
    int point_s = 0;
    for (i1=0; i1<n-1; i1++) {
        if (v[i1] < pivot) {
            swap(v, i1, point_s);
            swap(y, i1, point_s);
            point_s++;
        }
    }
    swap(v, point_s, n-1);
    swap(y, point_s, n-1);
    
    if (point_s == k) {
        return pivot;
    } else if (point_s > k) {
        return find_kth(v, point_s, k, y);
    } else {
        return find_kth(v + point_s + 1, n - point_s - 1, k - point_s - 1, y + point_s + 1);
    }
}


void swap(unsigned int array[], int i, int j) {
    int temp = array[i];
    array[i] = array[j];
    array[j] = temp;
}

int main(argc,argv)
int argc;
char *argv[];
{
    int num_quadrants;
    int  namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    double startwtime = 0.0, endwtime;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Get_processor_name(processor_name,&namelen);
    
    if (argc != 2)
    {
        fprintf (stderr, "Usage: recursive_bisection <#of quadrants>\n");
        MPI_Finalize();
        exit (0);
    }
    
    fprintf (stderr,"Process %d on %s\n", myid, processor_name);
    
    num_quadrants = atoi (argv[1]);
    
    if (myid == 0)
        fprintf (stdout, "Extracting %d quadrants with %d processors \n", num_quadrants, numprocs);
        
        if (myid == 0)
        {
            int i;
            
            srand (10000);
            
            for (i = 0; i < NUM_POINTS; i++)
                X_axis[i] = (unsigned int)rand();

            
            for (i = 0; i < NUM_POINTS; i++)
                Y_axis[i] = (unsigned int)rand();

            //start timer at process 0
            printf("\nComputing Parallely Using MPI.\n");
            startwtime = MPI_Wtime();
        }
    
    MPI_Bcast(&X_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Y_axis, NUM_POINTS, MPI_INT, 0, MPI_COMM_WORLD);  
    
    find_quadrants (num_quadrants);
     MPI_Barrier(MPI_COMM_WORLD);

    if (myid == 0) {
        endwtime = MPI_Wtime();
        printf("\nelapsed time = %f\n", endwtime - startwtime);
        printf("\nTotal cost:  %lf \n", global_cost);

    }
    
    MPI_Finalize();


    return 0;
}