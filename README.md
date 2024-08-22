# CUDA Utils

CUDA Utils is a header-only library that significantly simplifies complex CUDA kernel code. It provides intuitive wrapper classes for multi-dimensional tensors, making CUDA programming more readable and less error-prone, especially for advanced use cases like high-performance GEMM implementations.

## Usage Examples

The following examples demonstrate how CUDA Utils can dramatically improve code readability and reduce complexity in CUDA kernels. These snippets are based on real-world usage in high-performance GEMM kernels like [QuadMul](https://github.com/carsonpo/quadmul), [OctoMul](https://github.com/carsonpo/octomul), and [OctoQuadMul](https://github.com/carsonpo/octoquadmul).

### Example 1: Simplified Dynamic Indexing and Masking

#### Before:
```cpp
if (input_mask[batch_idx * num_heads * input_dim1 * input_dim2 +
               head_idx * input_dim1 * input_dim2 +
               mask_i * input_dim2 + mask_j] == 0) {
    output_tensor[batch_idx * num_heads * output_dim1 * output_dim2 +
                  head_idx * output_dim1 * output_dim2 +
                  i * output_dim2 + j] = -INFINITY;
}
```

#### After (with CUDA Utils):
```cpp
GMemTensor4D<float> output(output_tensor, batch_size, num_heads, output_dim1, output_dim2);
GMemTensor4D<int> mask(input_mask, batch_size, num_heads, input_dim1, input_dim2);

if (mask.get(batch_idx, head_idx, mask_i, mask_j) == 0) {
    output.set(batch_idx, head_idx, i, j, -INFINITY);
}
```

### Example 2: Simplified Memory Loading in GEMM Kernels

#### Before:
```cpp
uint8_t *shared_ptr = &shared_A[stage][row * Config::kTileSizeK + col];
uint8_t *global_ptr = &A[batch_idx * M * Config::K + 
                         (block_row_start + row) * Config::K + 
                         k_offset + col];
__pipeline_memcpy_async(shared_ptr, global_ptr, sizeof(Data128B));

```

#### After (with CUDA Utils):
```cpp
__pipeline_memcpy_async(
    smemA.get_ptr(stage, row, col),
    gmemA.get_ptr(batch_idx, block_row_start + row, k_offset + col),
    sizeof(Data128B));
```

## Benefits

1. **Improved Readability**: Complex indexing operations become self-explanatory.
2. **Reduced Errors**: Multi-dimensional index calculations are encapsulated, minimizing indexing errors.
3. **Performance-Oriented**: Designed for high-performance computing with efficient memory access patterns.
4. **Type-Safe Memory Reinterpretation**: `get_reinterpreted<>()` and `set_reinterpreted<>()` methods allow safe and easy reinterpretation of memory.
5. **Simplified Shared Memory Management**: Easier setup and access to shared memory in complex kernels.

## License

MIT