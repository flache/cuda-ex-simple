/**
 * demonstration of memory allocation needed to do parallel vector addition
 */

#include <stdio.h>
#include <stdlib.h>


__global__ void add(int* a, int* b, int* c) {

    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
    printf("%3d %14d %14d %14d \n",blockIdx.x, a[blockIdx.x], b[blockIdx.x], c[blockIdx.x]);
}

void random_ints(int *start, int num) {
    for (unsigned int i = 0; i < num; i++) {
        start[i] = rand();
    }
}


#define N 4
int main(void) {

    int *a, *b, *c; // host copies of a, b, c
    int *d_a, *d_b, *d_c; // device copies of a, b, c

    int size = N * sizeof(int);

    // Alloc space for device copies of a, b, c
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Alloc space for host copies of a, b, c and setup input values
    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

      // Copy inputs to device
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // Launch add() kernel on GPU with N blocks
    add<<<N,1>>>(d_a, d_b, d_c);

    // Copy result back to host
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    printf("kernel successfull finished \n");
    for (unsigned int i = 0; i < N; i++) {
        printf("%d %d \n", i, c[i]);
    }
    // Cleanup
    free(a); free(b); free(c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}
