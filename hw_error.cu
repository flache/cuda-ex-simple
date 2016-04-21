/**
 * demonstration cuda error handling
 */

#include <stdio.h>

__global__ void myhost(int *inp) {


    int b = inp[1];

    printf("b has value %d \n", b);
}

int main(void) {

    int host_inp[2] = {1,2};

    printf("Hello World! 123\n");

    myhost<<<1,1>>>(host_inp);

    //sync
    cudaError_t err = cudaDeviceSynchronize();
    printf("\nError: %s \n", cudaGetErrorString(err));
    return 0;
}
     
