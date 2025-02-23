#define TILE_DIM 32
#define BLOCK_ROWS 8
#define SECTION_SIZE 1024

__global__ void
SinglePassKoggeStoneScan(const unsigned int* input,
                         unsigned int* output,
                         const unsigned int length,
                         unsigned int* flags,
                         unsigned int* scanValue,
                         unsigned int* blockCounter)
{
    __shared__ unsigned int bid_s;
    __shared__ unsigned int XY[SECTION_SIZE];

    if (threadIdx.x == 0)
    {
        bid_s = atomicAdd(blockCounter, 1);
    }
    __syncthreads();

    const int bid = bid_s;
    const int idx = bid * blockDim.x + threadIdx.x;

    if (idx < length)
    {
        XY[threadIdx.x] = input[idx];
    }
    else
    {
        XY[threadIdx.x] = 0;
    }
    __syncthreads();

    for (int stride = 1; stride < SECTION_SIZE; stride *= 2)
    {
        __syncthreads();
        float tmp = 0;
        if (threadIdx.x >= stride)
        {
            tmp = XY[threadIdx.x] + XY[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride)
        {
            XY[threadIdx.x] = tmp;
        }
    }
    __syncthreads();

    __shared__ unsigned int previousSum;
    if (threadIdx.x == 0)
    {
        while (bid >= 1 && atomicAdd(&flags[bid], 0) == 0)
        {
        }
        previousSum = scanValue[bid];
        scanValue[bid + 1] = XY[blockDim.x - 1] + previousSum;
        __threadfence();
        atomicAdd(&flags[bid + 1], 1);
    }
    __syncthreads();

    if (idx < length)
    {
        output[idx] = XY[threadIdx.x] + previousSum;
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