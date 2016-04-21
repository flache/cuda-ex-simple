/*
 * Demonstration of 2-dimensional block- and thread-indices
 * mostly the same as vec_addition example.
 * adds up a square matrix with height and with N (Means N^2 calculations)
 * kernel is divided in block with THREADS_PER_BLOCK_X * THREADS_PER_BLOCK_X threads per block
 */

#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#define N 4
#define THREADS_PER_BLOCK_X 2

__global__ void add(int (*a)[N], int (*b)[N], int (*c)[N]) {

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    c[x][y] = a[x][y] + b[x][y];

	printf("%3d %3d %14d %14d %14d \n",x,y, a[x][y], b[x][y], c[x][y]);
}


void random_ints(int *start, int num) {
	for (unsigned int i = 0; i < num; i++) {
		start[i] = rand();
	}
}



int main(void) {

	int (*a)[N], (*b)[N], (*c)[N]; // host copies of a, b, c
	int (*d_a)[N], (*d_b)[N], (*d_c)[N]; // device copies of a, b, c

    //matrix size in bytes
	int size =  N *N*sizeof(int);

    //allcoate memory on the host
    a = (int (*)[N]) malloc(size);
    b = (int (*)[N]) malloc(size);
    c = (int (*)[N]) malloc(size);

    //allocate memory on device
    cudaMalloc((void ***)&d_a, size);
    cudaMalloc((void ***)&d_b, size);
    cudaMalloc((void ***)&d_c, size);


    //fill with random ints
    for (unsigned int i = 0; i < N; i++) {
        random_ints(a[i], N);
        random_ints(b[i], N);
    }


	// Alloc space for host copies of a, b, c and setup input values
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    //calculate number of blocks
    int num_blocks_x = (N / THREADS_PER_BLOCK_X);

	// Launch add() kernel on GPU with N blocks
	add<<< dim3(num_blocks_x,num_blocks_x) , dim3(THREADS_PER_BLOCK_X,THREADS_PER_BLOCK_X) >>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);



	printf("kernel successfull finished \n");
	for (unsigned int i = 0; i < N; i++) {
		for (unsigned int j = 0; j < N; j++) {

				printf("%d %d %d \n", i, j, c[i][j]);
		}
	}


    //cleanup
	free(a);
    free(b);
    free(c);
    cudaFree(d_a); 
    cudaFree(d_b); 
    cudaFree(d_c);

	return 0;
}
