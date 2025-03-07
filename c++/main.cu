/*
 * SPDX-License-Identifier: GPL-3.0-only
 *
 * Author: Alessio Bugetti <alessiobugetti98@gmail.com>
 */

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

double
SequentialIntegralImage(const unsigned int* input,
                        unsigned int* output,
                        const unsigned int height,
                        const unsigned int width)
{
    const auto start = std::chrono::high_resolution_clock::now();

    output[0] = input[0];

    for (unsigned int i = 1; i < height; i++)
    {
        output[i * width] = input[i * width] + output[(i - 1) * width];
    }

    for (unsigned int j = 1; j < width; j++)
    {
        output[j] = input[j] + output[j - 1];
    }

    for (unsigned int i = 1; i < height; i++)
    {
        for (unsigned int j = 1; j < width; j++)
        {
            output[i * width + j] = input[i * width + j] + output[(i - 1) * width + j] +
                                    output[i * width + j - 1] - output[(i - 1) * width + j - 1];
        }
    }
    const auto stop = std::chrono::high_resolution_clock::now();

    return std::chrono::duration<double, std::milli>(stop - start).count();
}

double
NaiveCudaIntegralImage(const unsigned int* hostInput,
                       unsigned int* hostOutput,
                       const unsigned int height,
                       const unsigned int width)
{
    const unsigned int pixelCount = height * width;
    const unsigned int imageSize = sizeof(unsigned int) * pixelCount;

    unsigned int *deviceInput, *deviceOutput;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInput), imageSize));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), imageSize));

    CHECK(cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice));

    dim3 blockScan(1, SECTION_SIZE);
    dim3 gridScan(1, (height + SECTION_SIZE - 1) / SECTION_SIZE);
    dim3 gridTransposedScan(1, (width + SECTION_SIZE - 1) / SECTION_SIZE);
    dim3 blockTranspose(TILE_DIM, BLOCK_ROWS);
    dim3 gridTranspose((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 gridTransposedTranspose(gridTranspose.y, gridTranspose.x);

    CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::high_resolution_clock::now();
    SumRows<<<gridScan, blockScan>>>(deviceInput, height, width);
    Transpose<<<gridTranspose, blockTranspose>>>(deviceInput, deviceOutput, height, width);
    SumRows<<<gridTransposedScan, blockScan>>>(deviceOutput, width, height);
    Transpose<<<gridTransposedTranspose, blockTranspose>>>(deviceOutput,
                                                           deviceInput,
                                                           width,
                                                           height);

    CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::high_resolution_clock::now();

    CHECK(cudaMemcpy(hostOutput, deviceInput, imageSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));

    return std::chrono::duration<double, std::milli>(stop - start).count();
}

double
CudaIntegralImage(const unsigned int* hostInput,
                  unsigned int* hostOutput,
                  const unsigned int height,
                  const unsigned int width)
{
    const unsigned int pixelCount = height * width;
    const unsigned int imageSize = sizeof(unsigned int) * pixelCount;

    unsigned int *deviceInput, *deviceOutput;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceInput), imageSize));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&deviceOutput), imageSize));
    CHECK(cudaMemcpy(deviceInput, hostInput, imageSize, cudaMemcpyHostToDevice));

    const unsigned int numBlocksPerRow = (width + SECTION_SIZE - 1) / SECTION_SIZE;
    const unsigned int lengthS = height * numBlocksPerRow;
    const size_t sizeS = sizeof(unsigned int) * lengthS;

    const unsigned int numBlocksPerRowTransposed = (height + SECTION_SIZE - 1) / SECTION_SIZE;
    const unsigned int lengthTransposedS = width * numBlocksPerRowTransposed;
    const size_t sizeTransposedS = sizeof(unsigned int) * lengthTransposedS;

    unsigned int *S, *flags, *blockCounter, *transposedS, *transposedFlags, *transposedBlockCounter;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&S), sizeS));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&flags), sizeS));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&transposedS), sizeTransposedS));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&transposedFlags), sizeTransposedS));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&blockCounter), sizeof(unsigned int)));
    CHECK(cudaMalloc(reinterpret_cast<void**>(&transposedBlockCounter), sizeof(unsigned int)));
    CHECK(cudaMemset(flags, 0, sizeS));
    CHECK(cudaMemset(S, 0, sizeS));
    CHECK(cudaMemset(blockCounter, 0, sizeof(unsigned int)));
    CHECK(cudaMemset(transposedFlags, 0, sizeTransposedS));
    CHECK(cudaMemset(transposedS, 0, sizeTransposedS));
    CHECK(cudaMemset(transposedBlockCounter, 0, sizeof(unsigned int)));

    dim3 blockScan(SECTION_SIZE);
    dim3 gridScan(numBlocksPerRow, height);
    dim3 gridTransposedScan(numBlocksPerRowTransposed, width);
    dim3 blockTranspose(TILE_DIM, BLOCK_ROWS);
    dim3 gridTranspose((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    dim3 gridTransposedTranspose(gridTranspose.y, gridTranspose.x);

    CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::high_resolution_clock::now();

    SinglePassRowWiseScan<<<gridScan, blockScan>>>(deviceInput,
                                                   deviceOutput,
                                                   flags,
                                                   S,
                                                   blockCounter,
                                                   height,
                                                   width);
    Transpose<<<gridTranspose, blockTranspose>>>(deviceOutput, deviceInput, height, width);
    SinglePassRowWiseScan<<<gridTransposedScan, blockScan>>>(deviceInput,
                                                             deviceOutput,
                                                             transposedFlags,
                                                             transposedS,
                                                             transposedBlockCounter,
                                                             width,
                                                             height);
    Transpose<<<gridTransposedTranspose, blockTranspose>>>(deviceOutput,
                                                           deviceInput,
                                                           width,
                                                           height);

    CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::high_resolution_clock::now();

    CHECK(cudaMemcpy(hostOutput, deviceInput, imageSize, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(deviceInput));
    CHECK(cudaFree(deviceOutput));
    CHECK(cudaFree(S));
    CHECK(cudaFree(flags));
    CHECK(cudaFree(transposedS));
    CHECK(cudaFree(transposedFlags));
    CHECK(cudaFree(blockCounter));
    CHECK(cudaFree(transposedBlockCounter));

    return std::chrono::duration<double, std::milli>(stop - start).count();
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

void
PrintImage(const unsigned int* input, const unsigned int height, const unsigned int width)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            std::cout << input[y * width + x] << " ";
        }
        std::cout << std::endl;
    }
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
        double totalNaiveCudaTime = 0;
        double totalOptimizedCudaTime = 0;

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            totalSequentialTime += SequentialIntegralImage(input, output, size, size);
            totalNaiveCudaTime += NaiveCudaIntegralImage(input, output, size, size);
            totalOptimizedCudaTime += CudaIntegralImage(input, output, size, size);
        }

        const double meanSequentialTime = totalSequentialTime / NUM_ITERATIONS;
        const double meanNaiveCudaTime = totalNaiveCudaTime / NUM_ITERATIONS;
        const double meanOptimizedCudaTime = totalOptimizedCudaTime / NUM_ITERATIONS;

        const double naiveSpeedup = meanSequentialTime / meanNaiveCudaTime;
        const double optimizedSpeedup = meanSequentialTime / meanOptimizedCudaTime;

        std::cout << std::setprecision(6);
        std::cout << "Image size: " << size << "x" << size << std::endl;
        std::cout << "Average Sequential Time: " << meanSequentialTime << " ms" << std::endl;
        std::cout << "Average Naive Kernel Time: " << meanNaiveCudaTime << " ms" << std::endl;
        std::cout << "Naive Kernel Speedup: " << naiveSpeedup << std::endl;
        std::cout << "Average Optimized Kernel Time: " << meanOptimizedCudaTime << " ms"
                  << std::endl;
        std::cout << "Optimized Kernel Speedup: " << optimizedSpeedup << std::endl;
        std::cout << "------------------------" << std::endl;

        delete[] input;
        delete[] output;
    }

    return 0;
}