1.

a.每个线程生成一个输出矩阵行的kernel函数

```c
__global__ void matrixMulKernelRow(float* C, const float* A, const float* B, int M, int N, int K) {
    // Calculate the row index of the element to be processed by this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Check if the row index is within the bounds of the output matrix
    if (row < M) {
        for (int col = 0; col < K; ++col) {
            float sum = 0;
            for (int i = 0; i < N; ++i) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}
```



执行配置参数:

```c++
// Host code to launch the kernel
void matrixMul(float* C, const float* A, const float* B, int M, int N, int K) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Define the number of threads per block and number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    matrixMulKernelRow<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, M, N, K);

    // Copy result from device to host
    cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

b.每个线程生成一个输出矩阵列的kernel函数以及执行配置参数

```c++
__global__ void matrixMulKernelCol(float* C, const float* A, const float* B, int M, int N, int K) {
    // Calculate the row index of the element to be processed by this thread
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the row index is within the bounds of the output matrix
    if (col < M) {
        for (int row = 0; row < M; ++row) {
            float sum = 0;
            for (int i = 0; i < N; ++i) {
                sum += A[row * N + i] * B[i * K + col];
            }
            C[row * K + col] = sum;
        }
    }
}
```

执行配置参数

```c++
// Host code to launch the kernel
void matrixMul(float* C, const float* A, const float* B, int M, int N, int K) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * N * sizeof(float));
    cudaMalloc(&d_B, N * K * sizeof(float));
    cudaMalloc(&d_C, M * K * sizeof(float));

    // Copy matrices from host to device
    cudaMemcpy(d_A, A, M * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * K * sizeof(float), cudaMemcpyHostToDevice);

    // Define the number of threads per block and number of blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (K + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    matrixMulKernelRow<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, M, N, K);

    // Copy result from device to host
    cudaMemcpy(C, d_C, M * K * sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

c.

按行处理通常受益于更自然的内存访问模式和索引简单性，在许多情况下可能会带来更好的性能。另一方面，按列处理可能更适合算法或操作，其结构天然围绕列输出或内存访问模式与列式数据结构良好对齐的情况。



2.

```c++
    __global__ void matrixMulKernelCol(float* A, const float* B, const float* C, int size) {
    // Calculate the row index of the element to be processed by this thread
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Check if the row index is within the bounds of the output matrix
    if (i < size) {
        float sum = 0;
        for (int j = 0; j < size; ++j) {
            sum += B[i * size + j] + C[j];
        }
        A[i] = sum;
    }
}
```



3.

a.每个block有多少个线程？ 512

b.每个grid有多少个线程？48640

c.grid中有多少个block？95

d.总共有多少个线程执行了代码中的第5行？45000



4.

1.如果矩阵是行优先存储的：20 * 400 + 10 = 8010

2.如果矩阵是列优先存储的： 10 * 500 + 20 = 5020



5.index=z×(width×height)+y×width+x

=5 * （400 * 500） + 20 * 400 + 10

=1008010
