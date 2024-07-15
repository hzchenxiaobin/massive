1.

a.每个线程生成一个输出矩阵行的kernel函数

```c
__global__ void matrixMulKernelRow(float* C, const float* A, const float* B, int M, int N, int K) {
    // Calculate the row index of the element to be processed by this thread
    int row = blockIdx.x * blockDim.x + threadIdx.x;

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

