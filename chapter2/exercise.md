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