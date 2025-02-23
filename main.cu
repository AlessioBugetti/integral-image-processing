#include "kernel.cu"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#define SEED 42
#define NUM_ITERATIONS 1000
#define CHECK(call)                                                                                \
    {                                                                                              \
        const cudaError_t error_code = call;                                                       \
        if (error_code != cudaSuccess)                                                             \
        {                                                                                          \
            fprintf(stderr, "CUDA Error:\n");                                                      \
            fprintf(stderr, "    File:       %s\n", __FILE__);                                     \
            fprintf(stderr, "    Line:       %d\n", __LINE__);                                     \
            fprintf(stderr, "    Error code: %d\n", error_code);                                   \
            fprintf(stderr, "    Error text: %s\n", cudaGetErrorString(error_code));               \
            exit(1);                                                                               \
        }                                                                                          \
    }

void
SequentialIntegralImage(const unsigned int* input,
                        unsigned int* output,
                        const unsigned int height,
                        const unsigned int width)
{
    for (unsigned int i = 0; i < height; i++)
    {
        unsigned int sum = 0;
        for (unsigned int j = 0; j < width; j++)
        {
            sum += input[i * width + j];
            output[i * width + j] = sum;
        }
    }

    for (unsigned int i = 0; i < height; i++)
    {
        for (unsigned int j = i + 1; j < width; j++)
        {
            std::swap(output[i * width + j], output[j * height + i]);
        }
    }

    for (unsigned int i = 0; i < width; i++)
    {
        unsigned int sum = 0;
        for (unsigned int j = 0; j < height; j++)
        {
            sum += output[i * height + j];
            output[i * height + j] = sum;
        }
    }

    for (unsigned int i = 0; i < width; i++)
    {
        for (unsigned int j = i + 1; j < height; j++)
        {
            std::swap(output[i * height + j], output[j * width + i]);
        }
    }
}

void
CudaIntegralImage(const unsigned int* hostInput,
                  unsigned int* hostOutput,
                  const unsigned int height,
                  const unsigned int width)
{
    const unsigned int pixelCount = height * width;
    const unsigned int imageSize = sizeof(unsigned int) * pixelCount;

    unsigned int S_len = (width + SECTION_SIZE - 1) / SECTION_SIZE;
    size_t size_S = sizeof(unsigned int) * S_len;

    unsigned int *deviceInput, *deviceOutput;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInput), imageSize));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), imageSize));

    CHECK(cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice));

    unsigned int *S, *flags, *blockCounter;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&S), size_S));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&flags), size_S));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&blockCounter), sizeof(unsigned int)));

    dim3 block(SECTION_SIZE);
    dim3 grid((width + SECTION_SIZE - 1) / SECTION_SIZE);

#pragma unroll
    for (int i = 0; i < height; i++)
    {
        CHECK(cudaMemset(blockCounter, 0, sizeof(unsigned int)));
        CHECK(cudaMemset(flags, 0, size_S));
        SinglePassKoggeStoneScan<<<grid, block>>>(deviceInput + i * width,
                                                  deviceOutput + i * width,
                                                  width,
                                                  flags,
                                                  S,
                                                  blockCounter);
    }

    block = dim3(TILE_DIM, BLOCK_ROWS);
    grid = dim3((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);

    Transpose<<<grid, block>>>(deviceOutput, deviceInput, height, width);

    S_len = (height + SECTION_SIZE - 1) / SECTION_SIZE;
    size_S = sizeof(unsigned int) * S_len;
    CHECK(cudaFree(S));
    CHECK(cudaFree(flags));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&S), size_S));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&flags), size_S));

    block = dim3(SECTION_SIZE);
    grid = dim3((height + SECTION_SIZE - 1) / SECTION_SIZE);

#pragma unroll
    for (int i = 0; i < width; i++)
    {
        CHECK(cudaMemset(flags, 0, size_S));
        CHECK(cudaMemset(blockCounter, 0, sizeof(unsigned int)));
        SinglePassKoggeStoneScan<<<grid, block>>>(deviceInput + i * width,
                                                  deviceOutput + i * width,
                                                  height,
                                                  flags,
                                                  S,
                                                  blockCounter);
    }

    block = dim3(TILE_DIM, BLOCK_ROWS);
    grid = dim3((height + TILE_DIM - 1) / TILE_DIM, (width + TILE_DIM - 1) / TILE_DIM);

    Transpose<<<grid, block>>>(deviceOutput, deviceInput, width, height);

    CHECK(cudaMemcpy(hostOutput, deviceInput, imageSize, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));
    CHECK(cudaFree(S));
    CHECK(cudaFree(flags));
}

unsigned int*
GenerateRandomGrayscaleImage(const unsigned int width, const unsigned int height)
{
    const unsigned int pixelCount = width * height;
    auto* output = new unsigned int[pixelCount];

    std::mt19937 generator(SEED);
    std::uniform_int_distribution distribution(0, 255);

    for (int i = 0; i < pixelCount; i++)
    {
        output[i] = distribution(generator);
    }

    return output;
}

int
main()
{
    const unsigned int sizes[] = {1024, 2048, 4096, 8192};

    // Warmup
    auto* outputWarmup = new unsigned int[1024 * 1024];
    const unsigned int* inputWarmup = GenerateRandomGrayscaleImage(1024, 1024);
    for (int i = 0; i < 10; i++)
    {
        CudaIntegralImage(inputWarmup, outputWarmup, 1024, 1024);
    }
    delete[] inputWarmup;
    delete[] outputWarmup;

    for (const auto& size : sizes)
    {
        const unsigned int* input = GenerateRandomGrayscaleImage(size, size);
        auto* output = new unsigned int[size * size];

        double totalSequentialTime = 0;
        double totalCudaTime = 0;

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            SequentialIntegralImage(input, output, size, size);
            auto stop = std::chrono::high_resolution_clock::now();
            totalSequentialTime += std::chrono::duration<double, std::milli>(stop - start).count();

            cudaDeviceSynchronize();
            start = std::chrono::high_resolution_clock::now();
            CudaIntegralImage(input, output, size, size);
            cudaDeviceSynchronize();
            stop = std::chrono::high_resolution_clock::now();
            totalCudaTime += std::chrono::duration<double, std::milli>(stop - start).count();
        }

        const double meanSequentialTime = totalSequentialTime / NUM_ITERATIONS;
        const double meanCudaTime = totalCudaTime / NUM_ITERATIONS;

        const double speedup = meanSequentialTime / meanCudaTime;

        std::cout << std::setprecision(6);
        std::cout << "Image size: " << size << "x" << size << std::endl;
        std::cout << "Average Sequential Time: " << meanSequentialTime << " ms" << std::endl;
        std::cout << "Average CUDA Time: " << meanCudaTime << " ms" << std::endl;
        std::cout << "Speedup: " << speedup << std::endl;
        std::cout << "------------------------" << std::endl;

        delete[] input;
        delete[] output;
    }

    return 0;
}