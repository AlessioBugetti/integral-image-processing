#
# SPDX-License-Identifier: GPL-3.0-only
#
# Author: Alessio Bugetti <alessiobugetti98@gmail.com>
#

import numpy as np
from numba import cuda, types
import time

SEED = 42
SECTION_SIZE = 256
TILE_DIM = 32
BLOCK_ROWS = 8
TILE_DIM_2 = TILE_DIM + 1
NUM_ITERATIONS = 1000


@cuda.jit
def single_pass_row_wise_scan(
        input, output, flags, scan_value, block_counter, num_rows, num_cols
):
    XY = cuda.shared.array(SECTION_SIZE, dtype=types.uint32)
    bid_s = cuda.shared.array(1, dtype=types.uint32)

    if cuda.threadIdx.x == 0:
        bid_s[0] = cuda.atomic.add(block_counter, 0, 1)
    cuda.syncthreads()

    bid = bid_s[0]
    blockIdx_x = bid // num_rows
    blockIdx_y = bid % num_rows
    col = blockIdx_x * SECTION_SIZE + cuda.threadIdx.x
    row = blockIdx_y

    pixel = row * num_cols + col

    if row < num_rows and col < num_cols:
        XY[cuda.threadIdx.x] = input[pixel]
    else:
        XY[cuda.threadIdx.x] = 0

    stride = 1
    while stride < SECTION_SIZE:
        cuda.syncthreads()
        tmp = 0
        if cuda.threadIdx.x >= stride:
            tmp = XY[cuda.threadIdx.x - stride]
        cuda.syncthreads()
        if cuda.threadIdx.x >= stride:
            XY[cuda.threadIdx.x] += tmp
        stride *= 2

    previous_sum = cuda.shared.array(1, dtype=types.uint32)
    if cuda.threadIdx.x == 0:
        while blockIdx_x >= 1 and cuda.atomic.add(flags, bid, 0) == 0:
            pass
        previous_sum[0] = scan_value[bid]
        scan_value[bid + num_rows] = XY[cuda.blockDim.x - 1] + previous_sum[0]
        cuda.threadfence()
        cuda.atomic.add(flags, bid + num_rows, 1)
    cuda.syncthreads()

    if row < num_rows and col < num_cols:
        output[pixel] = XY[cuda.threadIdx.x] + previous_sum[0]


@cuda.jit
def transpose(input, output, height, width):
    tile = cuda.shared.array((TILE_DIM, TILE_DIM_2), dtype=types.uint32)

    x = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.y

    for j in range(0, TILE_DIM, BLOCK_ROWS):
        if x < width and (y + j) < height:
            tile[cuda.threadIdx.y + j][cuda.threadIdx.x] = input[(y + j) * width + x]

    cuda.syncthreads()

    x = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.y

    for j in range(0, TILE_DIM, BLOCK_ROWS):
        if x < height and (y + j) < width:
            output[(y + j) * height + x] = tile[cuda.threadIdx.x][cuda.threadIdx.y + j]


def numba_integral_image(host_input, width, height):
    pixel_count = width * height
    host_output = np.zeros(pixel_count, dtype=np.uint32)

    device_input = cuda.to_device(host_input)
    device_output = cuda.device_array_like(host_output)

    num_blocks_per_row = (width + SECTION_SIZE - 1) // SECTION_SIZE
    length_s = height * num_blocks_per_row

    num_blocks_per_row_transposed = (height + SECTION_SIZE - 1) // SECTION_SIZE
    length_transposed_s = width * num_blocks_per_row_transposed

    S = cuda.device_array(length_s, dtype=np.uint32)
    flags = cuda.device_array(length_s, dtype=np.uint32)
    block_counter = cuda.device_array(1, dtype=np.uint32)
    transposed_S = cuda.device_array(length_transposed_s, dtype=np.uint32)
    transposed_flags = cuda.device_array(length_transposed_s, dtype=np.uint32)
    transposed_block_counter = cuda.device_array(1, dtype=np.uint32)

    flags[:] = 0
    S[:] = 0
    block_counter[:] = 0
    transposed_flags[:] = 0
    transposed_S[:] = 0
    transposed_block_counter[:] = 0

    grid_scan = (num_blocks_per_row, height, 1)
    block_scan = (SECTION_SIZE, 1, 1)
    grid_transposed_scan = (num_blocks_per_row_transposed, width, 1)
    block_transpose = (TILE_DIM, BLOCK_ROWS, 1)
    grid_transpose = (
        (width + TILE_DIM - 1) // TILE_DIM,
        (height + TILE_DIM - 1) // TILE_DIM,
        1,
    )
    grid_transposed_transpose = (
        (height + TILE_DIM - 1) // TILE_DIM,
        (width + TILE_DIM - 1) // TILE_DIM,
        1,
    )

    cuda.synchronize()
    start = time.perf_counter_ns()

    single_pass_row_wise_scan[grid_scan, block_scan](
        device_input, device_output, flags, S, block_counter, height, width
    )
    transpose[grid_transpose, block_transpose](
        device_output, device_input, height, width
    )
    single_pass_row_wise_scan[grid_transposed_scan, block_scan](
        device_input,
        device_output,
        transposed_flags,
        transposed_S,
        transposed_block_counter,
        width,
        height,
    )
    transpose[grid_transposed_transpose, block_transpose](
        device_output, device_input, width, height
    )

    cuda.synchronize()
    stop = time.perf_counter_ns()

    host_output = device_input.copy_to_host()

    return host_output, (stop - start) / 1000000


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
        numba_integral_image(hostInput, height_warmup, width_warmup)
        cuda.current_context().memory_manager.deallocations.clear()

    for size in sizes:
        hostInput = np.random.randint(0, 256, size=size * size, dtype=np.uint32)
        totalCudaTime = 0
        for iteration in range(NUM_ITERATIONS):
            hostOutput, iteration_time = numba_integral_image(hostInput, size, size)
            totalCudaTime += iteration_time
            cuda.current_context().memory_manager.deallocations.clear()

        meanCudaTime = totalCudaTime / NUM_ITERATIONS

        print(f"Image: {size}x{size}")
        print(f"Average CUDA Time: {meanCudaTime:.6g} ms")
        print("------------------------")
