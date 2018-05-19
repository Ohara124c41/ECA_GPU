#include "eeg.h"
#define BLOCK_SIZE 32
#define GRID_SIZE 8
double apen_correlation (int np, int32_t *x, unsigned int m, double r)
{
	bool set;
	unsigned int count;
	double sum = 0;

	for (unsigned int i = 0; i <= np - (m + 1) + 1; i++) {
		count = 0;
		for (unsigned int j = i; j <= np - (m + 1) + 1; j++) {
			set = false;

			for (unsigned int k = 0; k < m; k++) {
				if (abs(x[i + k] - x[j + k]) > r) {
					set = true;
					break;
				}
			}
			if (set == false) count= (i==j) ? count+1 : count+2;
		}
		sum += ((double) count) / ((double) np - m + 1);
	}

	return sum / ((double) np - m + 1);
}

__global__
void gpu_apen_correlation(int32_t *device_x, double *blocksums, int np, unsigned int m, double r)
{
	unsigned int i = blockIdx.y*blockDim.y+threadIdx.y;
	unsigned int j = blockIdx.x*blockDim.x+threadIdx.x;
	bool set=false;
	unsigned int index = threadIdx.y*BLOCK_SIZE+threadIdx.x;
	__shared__ int32_t counts[1024];   
	__syncthreads();
	if (i==j && i <= np - (m + 1) + 1 && j <= np - (m + 1) + 1)
		counts[index] = 1;
	else if (j < i)
		counts[index] = 0;
	else {
		for (unsigned int k = 0; k < m; k++) {
			if (abs(device_x[i + k] - device_x[j + k]) > r) {
				set = true;
				break;
				}
		}
		if (set == false && i <= np - (m + 1) + 1 && j <= np - (m + 1) + 1 )
			counts[index] = 2;
		else
			counts[index] = 0;
	}
	for (unsigned int s = 1; s < 1024; s *= 2) {
		if (index % (2 * s) == 0)
			counts[index] += counts[index + s];
		__syncthreads();
	}
	if(index==0)
	{
		blocksums[blockIdx.x*GRID_SIZE+blockIdx.y]=(double)(counts[0])/ (((double) np - m + 1)*((double) np - m + 1));
	}
	return;
}

void apen(int np, int32_t *x, float *a, unsigned int m, double r)
{
	// Based on: https://nl.mathworks.com/matlabcentral/fileexchange/26546-approximate-entropy
	float A;

	#ifdef CPU_ONLY
	A = log(apen_correlation(np, x, m, r) / apen_correlation(np, x, m + 1, r));

	#else
	// GPU CODE
	dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
	dim3 dimGrid(GRID_SIZE,GRID_SIZE);
	cudaError_t err;
	int32_t* device_x;
	err=cudaMalloc(&device_x, np*sizeof(int32_t));
	cudaCheckError(err);
	double* device_blocksums;
	err=cudaMalloc(&device_blocksums, GRID_SIZE*GRID_SIZE*sizeof(double));  
	cudaCheckError(err);
	err=cudaMemcpy(device_x, x, np*sizeof(int32_t), cudaMemcpyHostToDevice);
	cudaCheckError(err);
	gpu_apen_correlation<<<dimGrid, dimBlock>>>(device_x, device_blocksums, np, m, r);
	//cudaCheckError(cudaPeekAtLastError()); 
	double blocksums[GRID_SIZE*GRID_SIZE];
	err=cudaMemcpy(blocksums, device_blocksums, GRID_SIZE*GRID_SIZE*sizeof(double), cudaMemcpyDeviceToHost); 
	cudaCheckError(err);
	err=cudaFree(device_blocksums);
	cudaCheckError(err);
	int idxi;
	double sum = 0;
	for(idxi=0;idxi<GRID_SIZE*GRID_SIZE;idxi++){
		sum += ((double) blocksums[idxi]); 
	}
	err=cudaMalloc(&device_blocksums, GRID_SIZE*GRID_SIZE*sizeof(double));  
	cudaCheckError(err); 
	gpu_apen_correlation<<<dimGrid, dimBlock>>>(device_x, device_blocksums, np, m + 1, r);
	//cudaCheckError(cudaPeekAtLastError()); 
	err=cudaMemcpy(blocksums, device_blocksums, GRID_SIZE*GRID_SIZE*sizeof(double), cudaMemcpyDeviceToHost); 
	cudaCheckError(err);
	err=cudaFree(device_x);
	cudaCheckError(err);
	err=cudaFree(device_blocksums);
	cudaCheckError(err);
	double sum2 = 0;
	for(idxi=0;idxi<GRID_SIZE*GRID_SIZE;idxi++){
		sum2 += ((double) blocksums[idxi]); 
	}
	A = log(sum / sum2);
	#endif
	
	//Convert to fixed point
	*a = A;
}