#include <stdio.h>

#define TILE_SIZE 16

__global__ void mysgemm(int m, int n, int k, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A x B
     *   where A is a (m x k) matrix
     *   where B is a (k x n) matrix
     *   where C is a (m x n) matrix
     *
     * Use shared memory for tiling
     *
     ********************************************************************/
     __shared__ float A_tile[TILE_SIZE][TILE_SIZE];
    __shared__ float B_tile[TILE_SIZE][TILE_SIZE];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float Cvalue = 0;

    // Loop over tiles in A and B
    for (int p = 0; p < (k-1)/TILE_SIZE+1; p++)
    {
        // Load tile into shared memory
        if (row < m && p*TILE_SIZE+tx < k)
            A_tile[ty][tx] = A[row*k + p*TILE_SIZE+tx];
        else
            A_tile[ty][tx] = 0.0;
        if (p*TILE_SIZE+ty < k && col < n)
            B_tile[ty][tx] = B[(p*TILE_SIZE+ty)*n + col];
        else
            B_tile[ty][tx] = 0.0;
        __syncthreads();
        
        if (row < m && col < n)
        {
            for (int i = 0; i < TILE_SIZE; i++)
                Cvalue += A_tile[ty][i] * B_tile[i][tx];
        }
        __syncthreads();
    }
    if (row < m && col < n)
        C[row*n + col] = Cvalue;
    /*************************************************************************/
    // INSERT KERNEL CODE HERE
        
    /*************************************************************************/
}

void basicSgemm(int m, int n, int k, const float *A, const float *B, float *C)
{
    // Initialize thread block and kernel grid dimensions ---------------------


    const unsigned int BLOCK_SIZE = TILE_SIZE;

    //INSERT CODE HERE
    dim3 dim_grid, dim_block;

    dim_block.x = dim_block.y = BLOCK_SIZE; dim_block.z = 1;
    dim_grid.x = (n - 1) / BLOCK_SIZE + 1;
    dim_grid.y = (m - 1) / BLOCK_SIZE + 1;
    dim_grid.z = 1;

    // Invoke CUDA kernel -----------------------------------------------------

    //INSERT CODE HERE
    mysgemm<<<dim_grid, dim_block>>>(m, n, k, A, B, C);
	

}


