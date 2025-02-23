#define BLOCK_DIM 32
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
            // Attende i dati
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
    __shared__ float block[BLOCK_DIM][BLOCK_DIM + 1];

    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if ((xIndex < width) && (yIndex < height))
    {
        const unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = input[index_in];
    }

    __syncthreads();

    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if ((xIndex < height) && (yIndex < width))
    {
        const unsigned int index_out = yIndex * height + xIndex;
        output[index_out] = block[threadIdx.x][threadIdx.y];
    }
}