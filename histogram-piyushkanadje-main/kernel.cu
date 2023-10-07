#include <stdio.h>
#define BLOCK_SIZE 512
#define MAX_BLOCK_NUM 16
__global__ void histo_kernel(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins)
{

    __shared__ unsigned int privateHisto[4096];

    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
       privateHisto[i] = 0;
    __syncthreads();

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;

    while (i < num_elements)
    {
        atomicAdd(&(privateHisto[input[i]]), 1);
        i += stride;
    }
    __syncthreads();
    
    for (int i = threadIdx.x; i < num_bins; i += BLOCK_SIZE)
        atomicAdd(&(bins[i]), privateHisto[i]);

}

void histogram(unsigned int* input, unsigned int* bins, unsigned int num_elements, unsigned int num_bins) {

    dim3 dim_grid, dim_block;
    dim_block.x = BLOCK_SIZE; dim_block.y = dim_block.z = 1;
    int noBlock = (num_elements-1)/BLOCK_SIZE+1;
    dim_grid.x = (noBlock > MAX_BLOCK_NUM ? MAX_BLOCK_NUM : noBlock);
    dim_grid.y = dim_grid.z = 1;

    histo_kernel<<<dim_grid, dim_block>>>(input, bins, num_elements, num_bins);

	 
}


