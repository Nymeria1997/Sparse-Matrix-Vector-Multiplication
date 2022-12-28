#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#include <device_launch_parameters.h>
#define BlockDim 1024
#define ITER 3
#define warpSize 32
#define threadsPerBlock 64
#define sizeSharedMemory 1024
#define MAX_NUM_THREADS_PER_BLOCK 1024

char * FILENAME = "op_ibm32.txt";
typedef struct
{
  int n_row;
  int n_col;
  int *row_ptr;
  int *col_ind;
  double *val;
  int nnz;
}CSR;
void spmv_csr_scalar(CSR *csr,double *x ,double* y);
void spmv_csr_vector(CSR *csr,double *x ,double* y);
void spmv_csr_adaptive(CSR *csr,double *x ,double* y);
int spmv_csr_adaptive_rowblocks(int *ptr,int totalRows,int *rowBlocks);
void spmv_pcsr(CSR * csr,double *x,double *y) ;



//********************SpMV SCALAR**************************


__global__ void spmv_csr_scalar_kernel(double * d_val,double * d_vector,int * d_cols,int * d_ptr,int N, double * d_out)
{
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = tid; i < N; i += blockDim.x * gridDim.x)
  {
    double t = 0;
    int start = d_ptr[i];
    int end = d_ptr[i+1];
    // One thread handles all elements of the row assigned to it
    for (int j = start; j < end; j++)
    {
      int col = d_cols[j];
      t += d_val[j] * d_vector[col];
    }
    d_out[i] = t;
  }
}


void spmv_csr_scalar(CSR *csr,double *x ,double* y)
{
  double *d_vector,*d_val, *d_out;
  int *d_cols, *d_ptr;
  float time_taken;
  double gflop = 2 * (double) csr->nnz / 1e9;
  float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

	// Allocate memory on device
  cudaMalloc(&d_vector,csr->n_col*sizeof(double));
  cudaMalloc(&d_val,csr->nnz*sizeof(double));
  cudaMalloc(&d_out,csr->n_col*sizeof(double));
  cudaMalloc(&d_cols,csr->nnz*sizeof(int));
  cudaMalloc(&d_ptr,(csr->n_row+1)*sizeof(int));

	// Copy from host memory to device memory
  cudaMemcpy(d_vector,x,csr->n_col*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_val,csr->val,csr->nnz*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols,csr->col_ind,csr->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr,csr->row_ptr,(csr->n_row+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, csr->n_col*sizeof(double));

	// Run the kernel and time it
  cudaEventRecord(start);
  for (int i = 0; i < ITER; i++)
    spmv_csr_scalar_kernel<<<ceil(csr->n_row/(float)BlockDim),BlockDim>>>(d_val,d_vector,d_cols,d_ptr,csr->n_row,d_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

	// Copy from device memory to host memory 
  cudaMemcpy(y, d_out, csr->n_col*sizeof(double), cudaMemcpyDeviceToHost);

	// Free device memory
  cudaFree(d_vector);
  cudaFree(d_val);
  cudaFree(d_cols);
  cudaFree(d_ptr); 
  cudaFree(d_out);

	// Calculate and print out GFLOPs 
	
  time_taken = (milliseconds/ITER)/1000.0; 
  printf("Average time taken for %s is %f\n", "SpMV by GPU CSR Scalar Algorithm",time_taken);
  printf("Average GFLOP/s is %lf\n",gflop/time_taken);
	
}


//********************SpMV VECTOR**************************


__global__ void spmv_csr_vector_kernel(double * d_val,double * d_vector,int * d_cols,int * d_ptr,int N, double * d_out)
{
  // Thread ID in block
  int t = threadIdx.x;

  // Thread ID in warp
  int lane = t & (warpSize-1);

  // Number of warps per block
  int warpsPerBlock = blockDim.x / warpSize;

  // One row per warp
  int row = (blockIdx.x * warpsPerBlock) + (t / warpSize);

  __shared__ volatile double vals[BlockDim];

  if (row < N)
  {
    int rowStart = d_ptr[row];
    int rowEnd = d_ptr[row+1];
    double sum = 0;

    // Use all threads in a warp accumulate multiplied elements
    for (int j = rowStart + lane; j < rowEnd; j += warpSize)
    {
      int col = d_cols[j];
      sum += d_val[j] * d_vector[col];
    }
    vals[t] = sum;
    __syncthreads();

    // Reduce partial sums
    if (lane < 16) vals[t] += vals[t + 16];
    if (lane <  8) vals[t] += vals[t + 8];
    if (lane <  4) vals[t] += vals[t + 4];
    if (lane <  2) vals[t] += vals[t + 2];
    if (lane <  1) vals[t] += vals[t + 1];
    __syncthreads();

    // Write result
    if (lane == 0)
    {
      d_out[row] = vals[t];
    }
  }	
}

void spmv_csr_vector(CSR *csr,double *x ,double* y)
{
  double *d_vector,*d_val, *d_out;
  int *d_cols, *d_ptr;
  float time_taken;
  double gflop = 2 * (double) csr->nnz / 1e9;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Allocate memory on device
  cudaMalloc(&d_vector,csr->n_col*sizeof(double));
  cudaMalloc(&d_val,csr->nnz*sizeof(double));
  cudaMalloc(&d_out,csr->n_col*sizeof(double));
  cudaMalloc(&d_cols,csr->nnz*sizeof(int));
  cudaMalloc(&d_ptr,(csr->n_row+1)*sizeof(int));

  // Copy from host memory to device memory
  cudaMemcpy(d_vector,x,csr->n_col*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_val,csr->val,csr->nnz*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols,csr->col_ind,csr->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr,csr->row_ptr,(csr->n_row+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, csr->n_col*sizeof(double));

  // Run the kernel and time it
  cudaEventRecord(start);
  for (int i = 0; i < ITER; i++)
  spmv_csr_vector_kernel<<<ceil(csr->n_row / ((float)BlockDim/32)),BlockDim>>>(d_val,d_vector,d_cols,d_ptr,csr->n_row,d_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

	// Copy from device memory to host memory 
  cudaMemcpy(y, d_out, csr->n_col*sizeof(double), cudaMemcpyDeviceToHost);

	// Free device memory
  cudaFree(d_vector);
  cudaFree(d_val);
  cudaFree(d_cols);
  cudaFree(d_ptr); 
  cudaFree(d_out);

	// Calculate and print out GFLOPs  
	
  time_taken = (milliseconds/ITER)/1000.0; 
  printf("Average time taken for %s is %f\n", "SpMV by GPU CSR Vector Algorithm",time_taken);
  printf("Average GFLOP/s is %lf\n",gflop/time_taken);
}

//********************SpMV ADAPTIVE**************************

__global__ void spmv_csr_adaptive_kernel(double * d_val,double * d_vector,int * d_cols,int * d_ptr,int N, int * d_rowBlocks, double * d_out)
{
  int startRow = d_rowBlocks[blockIdx.x];
  int nextStartRow = d_rowBlocks[blockIdx.x + 1];
  int num_rows = nextStartRow -  startRow;
  int i = threadIdx.x;
  __shared__ volatile double LDS[BlockDim];
  // If the block consists of more than one row then run CSR Stream
  if (num_rows > 1) 
  {
    int nnz = d_ptr[nextStartRow] - d_ptr[startRow];
    int first_col = d_ptr[startRow];

    // Each thread writes to shared memory
    if (i < nnz)
    {
      LDS[i] = d_val[first_col + i] * d_vector[d_cols[first_col + i]];
    }
    __syncthreads();     

    // Threads that fall within a range sum up the partial results
    for (int k = startRow + i; k < nextStartRow; k += blockDim.x)
    {
      double temp = 0;
      for (int j= (d_ptr[k] - first_col); j < (d_ptr[k + 1] - first_col); j++)
      {
          temp = temp + LDS[j];
      }
      d_out[k] = temp;
    }
  }
  // If the block consists of only one row then run CSR Vector
  else 
  {
    // Thread ID in warp
    int rowStart = d_ptr[startRow];
    int rowEnd = d_ptr[nextStartRow];

    double sum = 0;

    // Use all threads in a warp to accumulate multiplied elements
    for (int j = rowStart + i; j < rowEnd; j += BlockDim)
    {
      int col = d_cols[j];
      sum += d_val[j] * d_vector[col];
    }

    LDS[i] = sum;
    __syncthreads();

    // Reduce partial sums
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) 
    {
      __syncthreads();
      if (i < stride)
        LDS[i] += LDS[i + stride]; 
    }
    // Write result
    if (i == 0)
      d_out[startRow] = LDS[i];
  }	
}


int spmv_csr_adaptive_rowblocks(int *ptr,int totalRows,int *rowBlocks)
{
  rowBlocks[0] = 0; 
  int sum = 0; 
  int last_i = 0; 
  int ctr = 1;
  for (int i = 1; i < totalRows; i++) 
  {
    // Count non-zeroes in this row 
    sum += ptr[i] - ptr[i-1];
    if (sum == BlockDim)
    {
      // This row fills up LOCAL_SIZE 
      last_i = i;
      rowBlocks[ctr++] = i;
      sum = 0;
    }
    else if (sum > BlockDim)
    {
      if (i - last_i > 1) 
      {
        // This extra row will not fit 
        rowBlocks[ctr++] = i - 1;
        i--;
      }
      else if (i - last_i == 1)
        // This one row is too large
        rowBlocks[ctr++] = i;
      last_i = i;
      sum = 0;
    }
  }
  rowBlocks[ctr++] = totalRows;
  return ctr;
}

void spmv_csr_adaptive(CSR *csr,double *x ,double* y)
{
  double *d_vector,*d_val, *d_out;
  int *d_cols, *d_ptr , *d_rowBlocks;
  float time_taken;
  double gflop = 2 * (double) csr->nnz / 1e9;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  int *rowBlocks = (int *) malloc(csr->n_row*sizeof(int));

  
  // Allocate memory on device
  cudaMalloc(&d_vector,csr->n_col*sizeof(double));
  cudaMalloc(&d_val,csr->nnz*sizeof(double));
  cudaMalloc(&d_out,csr->n_col*sizeof(double));
  cudaMalloc(&d_cols,csr->nnz*sizeof(int));
  cudaMalloc(&d_ptr,(csr->n_row+1)*sizeof(int));

  // Copy from host memory to device memory
  cudaMemcpy(d_vector,x,csr->n_col*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_val,csr->val,csr->nnz*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols,csr->col_ind,csr->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr,csr->row_ptr,(csr->n_row+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, csr->n_col*sizeof(double));

  /* Calculate the row blocks needed on the host, allocate memory on device 
  and copy contents of rowblocks array to device */
  int countRowBlocks = spmv_csr_adaptive_rowblocks(csr->row_ptr,csr->n_row,rowBlocks);

  cudaMalloc(&d_rowBlocks,countRowBlocks*sizeof(int));
  cudaMemcpy(d_rowBlocks,rowBlocks,countRowBlocks*sizeof(int),cudaMemcpyHostToDevice);

  // Run the kernel and time it
  cudaEventRecord(start);
  for (int i = 0; i < ITER; i++)
    spmv_csr_adaptive_kernel<<<(countRowBlocks-1),BlockDim>>>(d_val,d_vector,d_cols,d_ptr,csr->n_row,d_rowBlocks,d_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Copy from device memory to host memory 
  cudaMemcpy(y,d_out,csr->n_row*sizeof(double),cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_vector);
  cudaFree(d_val);
  cudaFree(d_cols);
  cudaFree(d_ptr);
  cudaFree(d_out);
  cudaFree(d_rowBlocks);
  free(rowBlocks);

  time_taken = (milliseconds/ITER)/1000.0;
  printf("Average time taken for %s is %f\n", "SpMV by GPU CSR Adaptive Algorithm",time_taken);
  printf("Average GFLOP/s is %lf\n",gflop/time_taken);
	   
}


//********************SpMV PCSR**************************



__global__ void spmv_pcsr_kernel1(double * d_val,double * d_vector,int * d_cols,int d_nnz, double * d_v)
{
    	int tid = blockIdx.x * blockDim.x + threadIdx.x;
    	int icr = blockDim.x * gridDim.x;
    	while (tid < d_nnz){
		d_v[tid] = d_val[tid] * d_vector[d_cols[tid]];
        	tid += icr;
    	}
}

__global__ void spmv_pcsr_kernel2(double * d_v,int * d_ptr,int N,double * d_out)
{
    	int gid = blockIdx.x * blockDim.x + threadIdx.x;
    	int tid = threadIdx.x;
    
    	__shared__ volatile int ptr_s[threadsPerBlock + 1];
    	__shared__ volatile double v_s[sizeSharedMemory];
 
   	// Load ptr into the shared memory ptr_s
    	ptr_s[tid] = d_ptr[gid];

	// Assign thread 0 of every block to store the pointer for the last row handled by the block into the last shared memory location
    	if (tid == 0) { 
    		if (gid + threadsPerBlock > N) {
	    		ptr_s[threadsPerBlock] = d_ptr[N];}
		else {
    	    		ptr_s[threadsPerBlock] = d_ptr[gid + threadsPerBlock];}
    	}
    	__syncthreads();

    	int temp = (ptr_s[threadsPerBlock] - ptr_s[0])/threadsPerBlock + 1;
    	int nlen = min(temp * threadsPerBlock,sizeSharedMemory);
    	double sum = 0;
    	int maxlen = ptr_s[threadsPerBlock];     
    	for (int i = ptr_s[0]; i < maxlen; i += nlen){
    		int index = i + tid;
    		__syncthreads();
    		// Load d_v into the shared memory v_s
    		for (int j = 0; j < nlen/threadsPerBlock;j++){
	    		if (index < maxlen) {
	        		v_s[tid + j * threadsPerBlock] = d_v[index];
	        		index += threadsPerBlock;
            		}
    		}
   	 	__syncthreads();

    		// Sum up the elements for a row
		if (!(ptr_s[tid+1] <= i || ptr_s[tid] > i + nlen - 1)) {
	   		int row_s = max(ptr_s[tid] - i, 0);
	    		int row_e = min(ptr_s[tid+1] -i, nlen);
	    		for (int j = row_s;j < row_e;j++){
				sum += v_s[j];
	    		}
		}	
    	}	
	// Write result
    	d_out[gid] = sum;
}

void spmv_pcsr(CSR * csr,double *x,double *y) 
{
  double *d_vector,*d_val, *d_out,*d_v;
  int *d_cols, *d_ptr;
  float time_taken;
  double gflop = 2 * (double) csr->nnz / 1e9;
  float milliseconds = 0;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	// Allocate memory on device
  cudaMalloc(&d_vector,csr->n_col*sizeof(double));
  cudaMalloc(&d_val,csr->nnz*sizeof(double));
  cudaMalloc(&d_v,csr->nnz*sizeof(double));
  cudaMalloc(&d_out,csr->n_col*sizeof(double));
  cudaMalloc(&d_cols,csr->nnz*sizeof(int));
  cudaMalloc(&d_ptr,(csr->n_row+1)*sizeof(int));

	// Copy from host memory to device memory
  cudaMemcpy(d_vector,x,csr->n_col*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_val,csr->val,csr->nnz*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols,csr->col_ind,csr->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr,csr->row_ptr,(csr->n_row+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, csr->n_col*sizeof(double));

	// Run the kernels and time them
  cudaEventRecord(start);
  for (int i = 0; i < ITER; i++) 
  {
    spmv_pcsr_kernel1<<<ceil(csr->nnz/(float)BlockDim),BlockDim>>>(d_val,d_vector,d_cols,csr->nnz,d_v);
    spmv_pcsr_kernel2<<<ceil(csr->n_row/(float)threadsPerBlock),threadsPerBlock>>>(d_v,d_ptr,csr->n_row,d_out);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);
   
	// Copy from device memory to host memory
  cudaMemcpy(y, d_out, csr->n_col*sizeof(double), cudaMemcpyDeviceToHost);

	// Free device memory
  cudaFree(d_vector);
  cudaFree(d_val);
  cudaFree(d_cols);
  cudaFree(d_ptr); 
  cudaFree(d_out);
  cudaFree(d_v);

  time_taken = (milliseconds/ITER)/1000.0; 
  printf("Average time taken for %s is %f\n", "SpMV by GPU PCSR Algorithm",time_taken);
  printf("Average GFLOP/s is %lf\n",gflop/time_taken);
	
}


//********************SpMV LIGHT**************************
template < int THREADS_PER_VECTOR, int MAX_NUM_VECTORS_PER_BLOCK>
__global__ void spmv_light_kernel(int* cudaRowCounter, int* d_ptr, int* d_cols,double* d_val, double* d_vector,int th_ct, double* d_out,int N) 
{
 // int THREADS_PER_VECTOR = th_ct;
 // int MAX_NUM_VECTORS_PER_BLOCK = MAX_NUM_THREADS_PER_BLOCK / th_ct;
	int i;
	double sum;
	int row;
	int rowStart, rowEnd;
	int laneId = threadIdx.x % THREADS_PER_VECTOR; //lane index in the vector
	int vectorId = threadIdx.x / THREADS_PER_VECTOR; //vector index in the thread block
	int warpLaneId = threadIdx.x & 31;	//lane index in the warp
	int warpVectorId = warpLaneId / THREADS_PER_VECTOR;	//vector index in the warp

	__shared__ volatile int space[MAX_NUM_VECTORS_PER_BLOCK][2];

	// Get the row index
	if (warpLaneId == 0) 
  {
		row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
	}
	// Broadcast the value to other threads in the same warp and compute the row index of each vector
	row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;
	
	while (row < N) 
  {

		// Use two threads to fetch the row offset
		if (laneId < 2) 
    {
			space[vectorId][laneId] = d_ptr[row + laneId];
		}
		rowStart = space[vectorId][0];
		rowEnd = space[vectorId][1];

		sum = 0;
		// Compute dot product
		if (THREADS_PER_VECTOR == 32) 
    {

			// Ensure aligned memory access
			i = rowStart - (rowStart & (THREADS_PER_VECTOR - 1)) + laneId;

			// Process the unaligned part
			if (i >= rowStart && i < rowEnd) 
      {
				sum += d_val[i] * d_vector[d_cols[i]];
			}

				// Process the aligned part
			for (i += THREADS_PER_VECTOR; i < rowEnd; i += THREADS_PER_VECTOR) 
      {
				sum += d_val[i] * d_vector[d_cols[i]];
			}
		} 
    else 
    {
			for (i = rowStart + laneId; i < rowEnd; i += THREADS_PER_VECTOR) 
      {
				sum += d_val[i] * d_vector[d_cols[i]];
			}
		}
		// Intra-vector reduction
		for (i = THREADS_PER_VECTOR >> 1; i > 0; i >>= 1) 
    {
			sum += __shfl_down_sync(0xffffffff,sum, i);
		}

		// Save the results
		if (laneId == 0) 
    {
			d_out[row] = sum;
		}

		// Get a new row index
		if(warpLaneId == 0)
    {
			row = atomicAdd(cudaRowCounter, 32 / THREADS_PER_VECTOR);
		}
		// Broadcast the row index to the other threads in the same warp and compute the row index of each vector
		row = __shfl_sync(0xffffffff,row, 0) + warpVectorId;

	}
}


void spmv_light(CSR * csr,double *x,double *y)
{
  double *d_vector,*d_val, *d_out;
  int *d_cols, *d_ptr;
  float time_taken;
  double gflop = 2 * (double) csr->nnz / 1e9;
  float milliseconds = 0;
  int meanElementsPerRow = csr->nnz/csr->n_row;
  int *cudaRowCounter;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

	// Allocate memory on device
  cudaMalloc(&d_vector,csr->n_col*sizeof(double));
  cudaMalloc(&d_val,csr->nnz*sizeof(double));
  cudaMalloc(&d_out,csr->n_col*sizeof(double));
  cudaMalloc(&d_cols,csr->nnz*sizeof(int));
  cudaMalloc(&d_ptr,(csr->n_row+1)*sizeof(int));
  cudaMalloc(&cudaRowCounter, sizeof(int));

  // Copy from host memory to device memory
  cudaMemcpy(d_vector,x ,csr->n_col*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_val,csr->val,csr->nnz*sizeof(double),cudaMemcpyHostToDevice);
  cudaMemcpy(d_cols,csr->col_ind,csr->nnz*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_ptr,csr->row_ptr,(csr->n_row+1)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemset(d_out, 0, csr->n_col*sizeof(double));
  cudaMemset(cudaRowCounter, 0, sizeof(int));

	// Choose the vector size depending on the NNZ/Row, run the kernel and time it
  cudaEventRecord(start);
  if (meanElementsPerRow <= 2) {
		for (int i = 0; i < ITER; i++) {
			spmv_light_kernel< 2, MAX_NUM_THREADS_PER_BLOCK / 2><<<ceil(csr->n_row/(float)BlockDim), BlockDim>>>(
				cudaRowCounter, d_ptr, d_cols,d_val,d_vector,2,d_out,csr->n_row);
			cudaMemset(cudaRowCounter, 0, sizeof(int));
		}
	} else if (meanElementsPerRow <= 4) {
		for (int i = 0; i < ITER; i++) {
			spmv_light_kernel<4, MAX_NUM_THREADS_PER_BLOCK / 4><<<ceil(csr->n_row/(float)BlockDim), BlockDim>>>(
				cudaRowCounter, d_ptr, d_cols,d_val, d_vector,4, d_out,csr->n_row);
			cudaMemset(cudaRowCounter, 0, sizeof(int));
		}
	} else if(meanElementsPerRow <= 64) {
		for (int i = 0; i < ITER; i++) {
			spmv_light_kernel< 8, MAX_NUM_THREADS_PER_BLOCK / 8><<<ceil(csr->n_row/(float)BlockDim), BlockDim>>>(
				cudaRowCounter,d_ptr,d_cols,d_val, d_vector,8, d_out,csr->n_row);
			cudaMemset(cudaRowCounter, 0, sizeof(int));
		}
	} else {
		for (int i = 0; i < ITER; i++){
			spmv_light_kernel<32, MAX_NUM_THREADS_PER_BLOCK / 32><<<ceil(csr->n_row/(float)BlockDim), BlockDim>>>(
				cudaRowCounter, d_ptr, d_cols,d_val, d_vector,32, d_out,csr->n_row);
			cudaMemset(cudaRowCounter, 0, sizeof(int));
		}
  }

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&milliseconds, start, stop);

  // Copy from device memory to host memory
  cudaMemcpy(y, d_out, csr->n_row*sizeof(double), cudaMemcpyDeviceToHost);
  
  // Free device memory	
  cudaFree(d_vector);
  cudaFree(d_val);
  cudaFree(d_cols);
  cudaFree(d_ptr);
  cudaFree(d_out);

  time_taken = (milliseconds/ITER)/1000.0;
  printf("Average time taken for %s is %f\n", "SpMV by GPU CSR LightSpMV Algorithm",time_taken);
  printf("Average GFLOP/s is %lf\n",gflop/time_taken);
	
}


CSR* Read_CSR(char *filename)
{
  CSR *csr = (CSR*)malloc(sizeof(CSR));
  
  FILE *f;
  int M, N, nz;   
  
  if ((f = fopen(filename, "r")) == NULL) 
    return NULL;

  fscanf(f, "%d %d %d",&N, &M, &nz); 
  
  csr->n_row = N;
  csr->n_col = M;
  csr->nnz = nz;
  csr->row_ptr = (int*)calloc((csr->n_row )+1,sizeof(int));
  csr->col_ind = (int*)calloc((csr->nnz) ,sizeof(int));
  csr->val = (double*)calloc((csr->nnz),sizeof(double));
  csr->row_ptr[(csr->n_row) ] = csr->nnz;

  printf("\nmatrix: %s N: %d M: %d nnz: %d\n",filename,csr->n_row,csr->n_col, csr->nnz);

  for(int i = 0; i< csr->nnz; i++)
  {
    fscanf(f,"%lf", &csr->val[i]);
  }
  for(int i = 0; i< csr->nnz; i++)
  {
    fscanf(f,"%d", &csr->col_ind[i]);
  }
  for(int i = 0; i< csr->n_row; i++)
  {
    fscanf(f,"%d", &csr->row_ptr[i]);
  }
  
  fclose(f);
  return csr;
}


int main() 
{
  CSR *csr ;

  csr = Read_CSR(FILENAME);
  
  if(csr==NULL)
    return 1;

  double *x = (double*)calloc(csr->n_col,sizeof(double));
  double *y = (double*)calloc(csr->n_col,sizeof(double));
  for(int i=0;i<csr->n_col;i++)
  {
    x[i]=1;
  }

  //sparse matrix vector multiplication
 
  clock_t start = clock();
  for(int iteration = 0 ; iteration < 1 ; iteration++)
  {  
    for(int i=0;i<csr->n_col;i++)
    {
      y[i] =0;
    }
    
    for(int i = 0; i< csr->n_col; i++)
    {
      for(int j=csr->row_ptr[i];j < csr->row_ptr[i+1];j++)
      {
        y[i] = y[i] + csr->val[j] * x[csr->col_ind[j]];
      }
    }
  }
  clock_t end = clock();
  for(int i=0;i<csr->n_col;i++)
  {
    x[i]=1;
  }
  free(y);
 
  y = (double*)calloc(csr->n_col,sizeof(double));
  spmv_csr_scalar(csr,x,y);
  free(y);
 
  y = (double*)calloc(csr->n_col,sizeof(double));
  spmv_csr_vector(csr,x,y);
  free(y);
 
  y = (double*)calloc(csr->n_col,sizeof(double));
  spmv_csr_adaptive(csr,x,y);
  free(y);
 
  y = (double*)calloc(csr->n_col,sizeof(double));
  spmv_pcsr(csr,x,y);
  free(y);
 
  y = (double*)calloc(csr->n_col,sizeof(double));
  spmv_light(csr,x,y);
  free(y);

  
  printf("\nTime required for serial execution: %lf s\n", (double)(end-start)/CLOCKS_PER_SEC );
  
  free(x);
  free(csr->col_ind);
  free(csr->row_ptr);
  free(csr->val);
  free(csr);
  return 0;
}