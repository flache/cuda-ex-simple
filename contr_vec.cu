/**
 * demonstration of shared memory. Kernel performs task of contracting consecutive vector elements. example:
 * RADIUS = 1 (means three consecutive numbers, (current, left and right), are added up)
 * input:   1   3   4   2   8   1   2 
 * output:  4   8   9   14  11  11  3
 */

#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 4
#define RADIUS 1
 // length of input array
#define N 16


__global__ void contr_vec_1d(int *inp, int *out) {

    __shared__ int block_mem[THREADS_PER_BLOCK + 2 * RADIUS];

    int glob_index = threadIdx.x + blockIdx.x * blockDim.x; 
    int loc_index = threadIdx.x + RADIUS;

    // Read input elements into shared memory
    block_mem[loc_index] = inp[glob_index]; 

    // read borders of current block
    if (threadIdx.x < RADIUS) {

        int before_index = glob_index - RADIUS;
        int after_index = glob_index + THREADS_PER_BLOCK;

        block_mem[loc_index - RADIUS] = (before_index < 0) ? 0 : inp[before_index];
        block_mem[loc_index + THREADS_PER_BLOCK] = (after_index >= N) ? 0 : inp[after_index]; 
    }

    // Synchronize (ensure all the data in block_mem is available)
    __syncthreads();

 
    // caluclate result
    int res = 0;
    for (int offset = -RADIUS ; offset <= RADIUS ; offset++){
        res += block_mem[loc_index + offset];
    }

    // Store the result
    out[glob_index] = res;
}



void random_ints(int *start, int num) {
    for (unsigned int i = 0; i < num; i++) {
        start[i] = rand()%10;
    }
}


int main(void) {

    int *inp, *out; //host cpies
    int *d_inp, *d_out; //device copies

    int size = N * sizeof(int);

    // Alloc space for device copies
    cudaMalloc((void **)&d_inp, size);
    cudaMalloc((void **)&d_out, size);

    // Alloc space for host copies of a, b, c and setup input values
    inp = (int *)malloc(size); random_ints(inp, N);
    out = (int *)malloc(size);

    // Copy inputs to device
    cudaMemcpy(d_inp, inp, size, cudaMemcpyHostToDevice);


    //numb_block should be integer!
    int num_blocks = N/THREADS_PER_BLOCK;


    // Launch add() kernel on GPU with N blocks
    contr_vec_1d<<<num_blocks,THREADS_PER_BLOCK>>>(d_inp, d_out);

    // Copy result back to host
    cudaMemcpy(out, d_out, size, cudaMemcpyDeviceToHost);


    printf("kernel successfull finished \n %5s %5s \n", "inp", "out");

    for (unsigned int i = 0; i < N; i++) {

        printf("%5d %5d \n", inp[i], out[i]);
    }
    // Cleanup
    free(inp); free(out);
    cudaFree(d_inp); cudaFree(d_out);
    return 0;
}
