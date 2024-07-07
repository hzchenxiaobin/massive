# 第二章 异构数据并行计算

### 章节大纲

---

**[2.1 数据并行](2.1.md)**

**[2.2 CUDA C程序结构](2.2.md)**

**[2.3 一个向量加法的kenel](2.3.md)**

**[2.4 global memory和数据传输](2.4.md)**

**[2.5 kernel函数和线程](2.5.md)**

**[2.6 调用内核函数](2.6.md)**

**[2.7 编译](2.7.md)**

**[2.8 总结](2.8.md)**

**[练习题](exercises.md)**

**[参考文献](references.md)**



数据并行性指的是在数据集的不同部分上执行的计算工作可以彼此独立地进行，因此可以并行执行。许多应用程序表现出丰富的数据并行性，使它们适合进行可伸缩的并行执行。因此，对并行程序员来说，熟悉数据并行性的概念以及编写利用数据并行性的代码的并行编程语言结构非常重要。在本章中，我们将使用CUDA C语言结构开发一个简单的数据并行程序。