import sys
import time as time
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

SEED = 42
SECTION_SIZE = 256
TILE_DIM = 32
BLOCK_ROWS = 8
NUM_ITERATIONS = 1

try:
    with open("../c++/kernel.cu", "r") as f:
        kernel_code = f.read()
except FileNotFoundError:
    print("Error: kernel.cu file not found")
    sys.exit(1)

mod = SourceModule(kernel_code)
SinglePassRowWiseScan = mod.get_function("SinglePassRowWiseScan")
Transpose = mod.get_function("Transpose")


def pycuda_integral_image(hostInput, width, height):
    pixelCount = width * height
    hostOutput = np.zeros(pixelCount, dtype=np.uint32)

    deviceInput = cuda.mem_alloc(hostInput.nbytes)
    deviceOutput = cuda.mem_alloc(hostOutput.nbytes)
    cuda.memcpy_htod(deviceInput, hostInput)

    numBlocksPerRow = (width + SECTION_SIZE - 1) // SECTION_SIZE
    lengthS = height * numBlocksPerRow
    sizeS = lengthS * np.uint32().nbytes

    numBlocksPerRowTransposed = (height + SECTION_SIZE - 1) // SECTION_SIZE
    lengthTransposedS = width * numBlocksPerRowTransposed
    sizeTransposedS = lengthTransposedS * np.uint32().nbytes

    S = cuda.mem_alloc(sizeS)
    flags = cuda.mem_alloc(sizeS)
    blockCounter = cuda.mem_alloc(np.uint32().nbytes)
    transposedS = cuda.mem_alloc(sizeTransposedS)
    transposedFlags = cuda.mem_alloc(sizeTransposedS)
    transposedBlockCounter = cuda.mem_alloc(np.uint32().nbytes)
    cuda.memset_d32(flags, 0, lengthS)
    cuda.memset_d32(S, 0, lengthS)
    cuda.memset_d32(blockCounter, 0, 1)
    cuda.memset_d32(transposedFlags, 0, lengthTransposedS)
    cuda.memset_d32(transposedS, 0, lengthTransposedS)
    cuda.memset_d32(transposedBlockCounter, 0, 1)

    gridScan = (numBlocksPerRow, height, 1)
    blockScan = (SECTION_SIZE, 1, 1)
    gridTransposedScan = (numBlocksPerRowTransposed, width, 1)
    blockTranpose = (TILE_DIM, BLOCK_ROWS, 1)
    gridTranspose = (
        (width + TILE_DIM - 1) // TILE_DIM,
        (height + TILE_DIM - 1) // TILE_DIM,
        1,
    )
    gridTransposedTranspose = (
        (height + TILE_DIM - 1) // TILE_DIM,
        (width + TILE_DIM - 1) // TILE_DIM,
        1,
    )

    cuda.Context.synchronize()
    start = time.perf_counter_ns()

    SinglePassRowWiseScan(
        deviceInput,
        deviceOutput,
        flags,
        S,
        blockCounter,
        np.uint32(height),
        np.uint32(width),
        block=blockScan,
        grid=gridScan,
    )
    Transpose(
        deviceOutput,
        deviceInput,
        np.uint32(height),
        np.uint32(width),
        block=blockTranpose,
        grid=gridTranspose,
    )
    SinglePassRowWiseScan(
        deviceInput,
        deviceOutput,
        transposedFlags,
        transposedS,
        transposedBlockCounter,
        np.uint32(width),
        np.uint32(height),
        block=blockScan,
        grid=gridTransposedScan,
    )
    Transpose(
        deviceOutput,
        deviceInput,
        np.uint32(width),
        np.uint32(height),
        block=blockTranpose,
        grid=gridTransposedTranspose,
    )

    cuda.Context.synchronize()
    stop = time.perf_counter_ns()

    cuda.memcpy_dtoh(hostOutput, deviceInput)

    deviceInput.free()
    deviceOutput.free()
    S.free()
    flags.free()
    blockCounter.free()
    transposedS.free()
    transposedFlags.free()
    transposedBlockCounter.free()

    return hostOutput, (stop - start) / 1000000


if __name__ == "__main__":
    np.random.seed(42)
    sizes = [1024, 2048, 4096, 8192]

    height_warmup = 1024
    width_warmup = 1024
    hostInput = np.random.randint(
        0, 256, size=height_warmup * width_warmup, dtype=np.uint32
    )

    # Warm-up
    for _ in range(10):
        pycuda_integral_image(hostInput, height_warmup, width_warmup)

    for size in sizes:
        hostInput = np.random.randint(0, 256, size=size * size, dtype=np.uint32)
        totalCudaTime = 0
        for iteration in range(NUM_ITERATIONS):
            hostOutput, iteration_time = pycuda_integral_image(hostInput, size, size)
            totalCudaTime += iteration_time

        meanCudaTime = totalCudaTime / NUM_ITERATIONS

        print(f"Image: {size}x{size}")
        print(f"Average CUDA Time: {meanCudaTime:.6g} ms")
        print("------------------------")
