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
    unsigned int blockIdx_x, blockIdx_y;

    if (width == height)
    {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    }
    else
    {
        const unsigned bid = blockIdx.x + gridDim.x * blockIdx.y;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    const unsigned int xIndexIn = blockIdx_x * TILE_DIM + threadIdx.x;
    const unsigned int yIndexIn = blockIdx_y * TILE_DIM + threadIdx.y;

    if (xIndexIn < width && yIndexIn < height)
    {
        const unsigned int index_in = xIndexIn + (yIndexIn * width);
        for (unsigned int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if (yIndexIn + i < height)
            {
                tile[threadIdx.y + i][threadIdx.x] = input[index_in + i * width];
            }
        }
    }

    __syncthreads();

    const unsigned int xIndexOut = blockIdx_y * TILE_DIM + threadIdx.x;
    const unsigned int yIndexOut = blockIdx_x * TILE_DIM + threadIdx.y;

    if (xIndexOut < height && yIndexOut < width)
    {
        for (unsigned int i = 0; i < TILE_DIM; i += BLOCK_ROWS)
        {
            if (xIndexOut < height && threadIdx.y + i < TILE_DIM)
            {
                const unsigned int index_out = xIndexOut + (yIndexOut * height);
                output[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
            }
        }
    }
}