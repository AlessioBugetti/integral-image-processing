#include <iomanip>
#include <iostream>
#include <random>

#define SEED 42
#define NUM_ITERATIONS 2

unsigned char*
GenerateRandomGrayscaleImage(const unsigned int width, const unsigned int height)
{
    const unsigned int pixelCount = width * height;
    auto* output = new unsigned char[pixelCount];

    std::mt19937 generator(SEED);
    std::uniform_int_distribution<int> distribution(0, 255);

    for (int i = 0; i < pixelCount; i++)
    {
        output[i] = static_cast<unsigned char>(distribution(generator));
    }

    return output;
}

void
SequentialIntegralImage(const unsigned char* input,
                        unsigned char* output,
                        const unsigned int height,
                        const unsigned int width)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            const unsigned int index = y * width + x;
            const unsigned int pixelValue = input[index];

            const unsigned int left = (x > 0) ? output[index - 1] : 0;
            const unsigned int top = (y > 0) ? output[index - width] : 0;
            const unsigned int topLeft = (x > 0 && y > 0) ? output[index - width - 1] : 0;

            output[index] = pixelValue + left + top - topLeft;
        }
    }
}

void
PrintImage(const unsigned char* input, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            std::cout << static_cast<int>(input[y * width + x]) << "\t";
        }
        std::cout << std::endl;
    }
}

int
main()
{
    const unsigned int sizes[] = {1024, 2048, 4096, 8192};

    for (const auto& size : sizes)
    {
        const unsigned char* input = GenerateRandomGrayscaleImage(size, size);
        auto* output = new unsigned char[size * size];

        double totalSequentialTime = 0;

        for (int i = 0; i < NUM_ITERATIONS; i++)
        {
            auto start = std::chrono::high_resolution_clock::now();
            SequentialIntegralImage(input, output, size, size);
            auto stop = std::chrono::high_resolution_clock::now();
            totalSequentialTime += std::chrono::duration<double, std::milli>(stop - start).count();
        }

        const double meanSequentialTime = totalSequentialTime / NUM_ITERATIONS;

        std::cout << std::setprecision(6);
        std::cout << "Image size: " << size << "x" << size << std::endl;
        std::cout << "Average Sequential Time: " << meanSequentialTime << " ms" << std::endl;
        // std::cout << "Average CUDA Time: " << meanCudaTime << " ms"
        //           << std::endl;
        // std::cout << "Speedup: " << speedup << std::endl;
        std::cout << "------------------------" << std::endl;

        delete[] input;
        delete[] output;
    }

    return 0;
}