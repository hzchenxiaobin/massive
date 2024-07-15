### 练习题

1.在本章中，我们实现了一个矩阵乘法核函数，使每个线程生成一个输出矩阵元素。在这个问题中，你将实现不同的矩阵-矩阵乘法核函数并进行比较。

a.编写一个让每个线程生成一个输出矩阵行的kernel函数。填写该设计的执行配置参数。

b.编写一个让每个线程生成一个输出矩阵列的kernel函数。填写该设计的执行配置参数。

c.分析这两种核函数设计的优缺点。



2.矩阵-向量乘法接收一个输入矩阵 B 和一个向量 C，并产生一个输出向量 A。输出向量 A 的每个元素是输入矩阵 B 的一行与向量 C 的点积，即
$$
A[i] = \sum ^jB[i][j] + C[j]
$$
为简单起见，我们只处理元素为单精度浮点数的方阵。请编写一个矩阵-向量乘法的 CUDA  kernel函数和可以通过四个参数调用的主机存根函数：输出矩阵的指针、输入矩阵的指针、输入向量的指针以及每个维度中的元素数量。每个线程用于计算一个输出向量元素。



3.思考下面的CUDA kernel函数和调用kernel函数的host函数：

```c++
__global__ void foo_kernel(float* a, float* b, unsigned int M, unsigned int N) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        b[row * N + col] = a[row * N + col] / 2.1f + 4.8f;
    }
}

void foo(float* a_d, float* b_d) {
    unsigned int M = 150;
    unsigned int N = 300;
    dim3 bd(16, 32);
    dim3 gd((N - 1) / 16 + 1, (M - 1) / 32 + 1);//(19, 5)
    foo_kernel<<<gd, bd>>>(a_d, b_d, M, N);
}
```

a.每个block有多少个线程？ 

b.每个grid有多少个线程？

c.grid中有多少个block？

d.总共有多少个线程执行了代码中的第5行？



4.有一个宽度为400高度为500的2D矩阵，该矩阵是存在一维数组中，算出row=20 col=10的元素在该数组中的索引值：

1.如果矩阵是行优先存储的。

2.如果矩阵是列优先存储的。



5.考虑一个三维张量，宽度为400，高度为500，深度为300。该张量按行优先顺序存储为一维数组。请指定张量在 x = 10，y = 20，z = 5 处的数组索引。





