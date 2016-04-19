/**
 * demonstration of asynchronous program execution
 */

#include <stdio.h>

__global__ void myhost(void) {

	printf("Hello World from the host\n");
}

int main(void) {

	printf("Hello World! 123\n");
	myhost<<<1,1>>>();
	//sync threads
	cudaDeviceSynchronize();
	return 0;
}
	 
