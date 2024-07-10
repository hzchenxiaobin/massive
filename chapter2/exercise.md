1.如果我们想使用grid中的每个线程来计算向量加法的一个输出元素，如何将线程/block索引映射到数据索引的表达式是什么？

(A) i=threadIdx.x + threadIdx.y;

(B) i=blockIdx.x + threadIdx.x;

(C) i=blockIdx.x * blockDim.x + threadIdx.x;

(D) i=blockIdx.x * threadIdx.x;



2.假设我们希望使用每个线程来计算向量加法的两个相邻元素。将线程/块索引映射到线程要处理的第一个元素的数据索引i的表达式是什么？

(A) i=blockIdx.x * blockDim.x + threadIdx.x +2;

(B) i=blockIdx.x * threadIdx.x 2;

(C) i=(blockIdx.x * blockDim.x + threadIdx.x) 2;

(D) i=blockIdx.x * blockDim.x 2 + threadIdx.x;



3.假设我们希望使用每个线程来计算向量加法中的两个元素。每个block处理 2⋅blockDim.x个连续的元素，这些元素形成两个部分。每个block中的所有线程首先处理一个部分，每个线程处理一个元素。然后它们将全部移动到下一个部分，每个线程处理一个元素。假设变量 ***i*** 应该是线程要处理的第一个元素的索引。将线程/block索引映射到第一个元素的数据索引的表达式是什么？

(A) i=blockIdx.x * blockDim.x + threadIdx.x +2;

(B) i=blockIdx.x * threadIdx.x 2;

(C) i=(blockIdx.x * blockDim.x + threadIdx.x) * 2;

(D) i=blockIdx.x * blockDim.x 2 + threadIdx.x;



4.对于向量加法，假设向量长度为8000，每个线程计算一个输出元素，一个block有1024个线程。程序员配置内核调用，以覆盖所有输出元素所需的最小block数。grid中将有多少线程？

(A) 8000

(B) 8196

(C) 8192

(D) 8200



5.如果我们想在CUDA global memory中分配一个包含 ***v*** 个整数元素的数组，cudaMalloc调用的第二个参数的合适表达式是什么？

(A) n

(B) v

(C) n * sizeof(int)

(D) v * sizeof(int)



6.如果我们想要在CUDA device上分配一个包含 ***n***个浮点数元素的数组，并且有一个浮点数指针变量 `A_d` 指向分配的内存，那么在 `cudaMalloc()` 调用的第一个参数中，适当的表达式是什么？

(A) n

(B) (void  *) A_d

(C)  *A_d

(D) (void  **) &A_d



7.如果我们希望将来自host数组 `A_h` 的3000字节数据（其中 `A_h` 是源数组第0个元素的指针）复制到device数组 `A_d`（其中 `A_d` 是目标数组第0个元素的指针），在CUDA中进行这种数据复制的适当API调用是什么？

(A) cudaMemcpy(3000, A_h, A_d, cudaMemcpyHostToDevice);

(B) cudaMemcpy(A_h, A_d, 3000, cudaMemcpyDeviceTHost);

(C) cudaMemcpy(A_d, A_h, 3000, cudaMemcpyHostToDevice);

(D) cudaMemcpy(3000, A_d, A_h, cudaMemcpyHostToDevice);



8.如何声明一个变量 `err`，使其能够适当接收CUDA API调用的返回值？

(A) int err;

(B) cudaError err;

(C) cudaError_t err;

(D) cudaSuccess_t err;



9.以下的CUDA核函数及其对应的调用它的主机函数：

```c
01 	__global__ void foo_kernel(float* a, float* b, unsigned int N){
02 	unsigned int i=blockIdx.x * blockDim.x + threadIdx.x;
03 	if(i < N) {
04 		b[i]=2.7f * a[i] - 4.3f;
05    }
06 	}

07 	void foo(float* a_d, float* b_d) {
08 		unsigned int N=200000;
09 		foo_kernel<<<(N + 128, 1)/128, 128>>>(a_d, b_d, N);
10 }
```

a.每个block中有多少个线程？

b.grid中有多少个线程？

c.grid中有多少个block？

d.有多少线程执行了第2行代码？

e.有多少线程执行了第4行代码？



10.一个新的暑期实习生对CUDA感到沮丧。他一直在抱怨CUDA非常繁琐。他不得不声明许多函数，他计划在host和device上执行两次，一次作为host函数，一次作为device函数。你的回应是什么？