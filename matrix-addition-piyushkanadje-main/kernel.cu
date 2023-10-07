#include <stdio.h>

#define TILE_SIZE 16

__global__ void matAdd(int dim, const float *A, const float *B, float* C) {

    /********************************************************************
     *
     * Compute C = A + B
     *   where A is a (dim x dim) matrix
     *   where B is a (dim x dim) matrix
     *   where C is a (dim x dim) matrix
     *
     ********************************************************************/

    
         int row = threadIdx.x+blockIdx.x * blockDim.x ;
    
	if(row<dim){
		for (int i=0;i<dim;i++){
			C[row+i*dim]=A[row+i*dim]+B[row+i*dim];
        }
    }
    

}

void basicMatAdd(int dim, const float *A, const float *B, float *C)
{
    

    const unsigned int BLOCK_SIZE = TILE_SIZE;
	
   
    dim3 DimGrid((dim-1)/BLOCK_SIZE +1,(dim-1)/BLOCK_SIZE +1, 1);
    dim3 DimBlock(BLOCK_SIZE , BLOCK_SIZE , 1);

    matAdd<<<DimGrid,DimBlock>>>(dim,A,B,C);
   
	
	// Invoke CUDA kernel -----------------------------------------------------

    

}

