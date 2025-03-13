/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Author: Alessio Bugetti <alessiobugetti98@gmail.com>
 */

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define SECTION_SIZE 256

__global__ void
SumRows(unsigned int* input, const unsigned int height, const unsigned int width)
{
    const unsigned int tid = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid < height)
    {
        for (int j = 1; j < width; j++)
        {
            input[tid * width + j] += input[tid * width + (j - 1)];
        }
    }
}

__global__ void
SinglePassRowWiseScan(const unsigned int* input,
                      unsigned int* output,
                      unsigned int* flags,
                      unsigned int* scanValue,
                      unsigned int* blockCounter,
                      const unsigned int numRows,
                      const unsigned int numCols)
{
    __shared__ unsigned int XY[SECTION_SIZE];
    __shared__ unsigned int bid_s;

    if (threadIdx.x == 0)
    {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();

    const unsigned int bid = bid_s;
    const unsigned int blockIdx_x = bid / numRows;
    const unsigned int blockIdx_y = bid % numRows;
    const int col = blockIdx_x * SECTION_SIZE + threadIdx.x;
    const unsigned int row = blockIdx_y;

    const int pixel = row * numCols + col;

    if (row < numRows && col < numCols)
    {
        XY[threadIdx.x] = input[pixel];
    }
    else
    {
        XY[threadIdx.x] = 0;
    }

    for (int stride = 1; stride < SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        unsigned int tmp = 0;
        if (threadIdx.x >= stride)
        {
            tmp = XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride)
        {
            XY[threadIdx.x] += tmp;
        }
    }
    __syncthreads();
    
    __shared__ unsigned int previousSum;
    if (threadIdx.x == 0)
    {
        while (blockIdx_x >= 1 && atomicAdd(&flags[bid], 0) == 0)
        {
        }
        previousSum = scanValue[bid];
        scanValue[bid + numRows] = XY[blockDim.x - 1] + previousSum;
        __threadfence();
        atomicAdd(&flags[bid + numRows], 1);
    }
    __syncthreads();

    if (row < numRows && col < numCols)
    {
        output[pixel] = XY[threadIdx.x] + previousSum;
    }
}

__global__ void
Transpose(const unsigned int* input,
          unsigned int* output,
          const unsigned int height,
          const unsigned int width)
{
    __shared__ unsigned int tile[TILE_DIM][TILE_DIM + 1];

    unsigned int x = blockIdx.x * TILE_DIM + threadIdx.x;
    unsigned int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if ((x < width) && (y + j < height))
        {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (unsigned int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        if ((x < height) && (y + j < width))
        {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}