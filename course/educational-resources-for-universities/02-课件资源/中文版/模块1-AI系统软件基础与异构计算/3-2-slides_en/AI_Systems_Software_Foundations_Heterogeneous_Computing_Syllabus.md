# 模块一：AI 系统软件基础与异构计算 (Module I: AI Systems Software Foundations & Heterogeneous Computing: Syllabus)

## 课程基本信息

- **课程编号**：04832240  
- **学时**：9 学时  
- **授课学期**：2026 年秋季学期  
- **上课时间**：（详见选课系统）  
- **上课地点**：（详见选课系统）  
- **授课单位**：北京大学计算机学院  
- **适用专业**：计算机科学与技术、软件工程、人工智能  
- **课程性质**：专业选修（系统结构与并行计算类组）

## 授课教师

- **主讲教师**：罗国杰  
- **电子邮件**：（待定）  
- **Office Hours**：（待定，欢迎通过邮件预约）

## 课程简介

本子课程聚焦人工智能时代异构计算的基础理论与编程实践。课程从 CPU/GPU/NPU 硬件体系结构对比出发，系统讲授 SIMT 编程模型、深度学习框架调用链路以及 Triton 语言基础，并通过向量加法、矩阵乘法等经典算子的实现，帮助同学们建立从硬件到软件栈的完整认知体系。课程以 FlagOS 生态为工程背景，强调理论与实践并重，为后续高级模块奠定坚实基础。

## 课程平台与在线实验环境

### FlagOS 智算系统软件栈

本课程的实验环境基于 FlagOS 智算系统软件栈。FlagOS 是由北京智源人工智能研究院（BAAI）主导开发的统一异构计算开源软件栈，面向多元计算架构构建统一开源技术栈，实现「一次开发、多芯复用、全域部署」。FlagOS 包含多个核心组件，包括：

- **FlagScale**：分布式训练/推理框架，支持跨异构硬件的大模型训练与部署  
- **FlagGems**：通用算子库，基于 Triton 语言实现的高性能 AI 算子集合  
- **FlagTree**：统一编译器，支持多 AI 芯片后端的 Triton 编译  
- **FlagCX**：跨芯片通信库，支持异构计算环境下的高性能集合通信  

课程实验中涉及的 Triton Kernel 开发、算子优化、MLIR Pass 编写、分布式训练部署等任务，均将在 FlagOS 生态环境中完成。

### 在线实验平台（Online Laboratory）

课程提供基于云端的在线实验平台，同学们无需自行配置本地 GPU 环境，即可通过浏览器远程访问 GPU 计算资源，完成全部课程实验。

**平台申请与登录**：  
开课后，教师将统一为选课同学开通在线实验平台账号。请同学们登录以下地址访问平台：  
在线实验平台入口：<https://flagos.io/OnlineLab>  
账号开通后，你将获得分配的 GPU 计算资源，可以直接在线编写、运行和调试代码。请注意合理使用计算时长，避免长时间占用资源而不释放。

**平台使用说明**：  
在线实验平台的详细使用文档（环境配置、实验提交流程、常见问题等）请参考：  
平台文档：<https://docs.flagos.io/projects/onlinelaboratory/en/latest/onlinelaboratory.html>  
建议同学们在第一次实验前仔细阅读以上文档，熟悉平台的基本操作流程。如果在使用过程中遇到技术问题，请先查阅文档，再通过课程讨论区或 Office Hours 寻求帮助。

**注意事项**：  

- 在线实验平台的计算资源专供本课程实验使用，请勿用于与课程无关的个人项目。  
- 实验完成后请及时保存代码和数据到本地，平台实例可能会定期回收。  
- 如果在高峰时段无法获取资源，请错峰使用，并合理规划实验时间。

### 课程相关链接汇总

以下是课程学习中你可能会用到的重要链接：

| 资源 | 网址 |
|------|------|
| FlagOS 官网 | <https://flagos.io> |
| 在线实验平台 | <https://flagos.io/OnlineLab> |
| 平台使用文档 | <https://docs.flagos.io/projects/onlinelaboratory/en/latest/onlinelaboratory.html> |
| FlagOS 开发者社区 | <https://flagos.io> |
| FlagGems GitHub | <https://github.com/FlagOpen/FlagGems> |
| FlagTree GitHub | <https://github.com/flagos-ai/FlagTree> |
| FlagCX GitHub | <https://github.com/flagos-ai/FlagCX> |

## 先修要求

选修本课程前，你应当具备以下基础：

- **计算机组成原理**：理解基本的处理器结构、存储层次与指令执行流程。  
- **操作系统原理**：了解进程/线程管理、内存管理等基本概念。  
- **深度学习导论**：熟悉神经网络基本概念，有使用 PyTorch 或 TensorFlow 的经验。  
- **C/C++ 程序设计**：具备较扎实的 C/C++ 编程能力。  

如果你在 CUDA、Triton 或 GPU 编程方面没有经验，不必担心——课程会从基础讲起。但课程节奏较快，建议提前熟悉 Python 和 PyTorch 的基本用法。

## 课程目标

完成本模块学习后，你将能够：

- 理解 CPU、GPU、NPU 等 AI 硬件的体系结构特征与设计权衡，认识摩尔定律放缓对计算系统带来的挑战。  
- 掌握 SIMT 编程模型、线程/束概念以及内存合并等异构计算核心编程技术。  
- 理解深度学习框架（PyTorch/TensorFlow）从 Python 前端到硬件执行的完整调用链路。  
- 掌握 Triton 语言基本语法，能够编写向量加法、矩阵乘法、Softmax 等基础算子。  
- 熟悉 FlagOS 生态与 FlagGems 算子库，具备在 FlagGems 框架下开发与测试算子的基本能力。  
- 理解内存层次结构优化的基本策略，包括 Shared Memory 使用、Bank Conflict 识别与 Tiling 技术。

## 课时安排

| 课时 | 主题与内容 |
|------|------------|
| 1 | 课程导论与 AI 硬件全景：CPU/GPU/NPU 体系结构对比，FlagOS 诞生背景 |
| 2 | 异构计算编程模型：SIMT 模型、线程与束概念、内存合并 |
| 3 | 深度学习框架调用链路分析：Python → C++ → CUDA Driver → Hardware |
| 4 | Triton 语言入门：Kernel 函数定义、Grid 与 Block 概念 |
| 5 | 环境搭建与工具链：Docker/Conda 配置、FlagOS 开发环境搭建 |
| 6 | 向量计算基础：用 Triton 实现 VectorAdd，理解 tl.load/tl.store |
| 7 | 矩阵乘法优化 I：Tiled Matrix Multiplication 原理与实现 |
| 8 | 内存层次结构优化：Shared Memory、Bank Conflict、Cache 局部性 |
| 9 | 模块期中项目：使用 Triton 实现 Softmax Kernel |

## 教材与参考资料

本课程没有指定的必修教材。阅读材料以课堂讲义、论文和在线文档为主。以下为推荐的参考资源：

### 推荐参考书

- 郑纬民，张晨曦，等.《智能计算系统》. 清华大学出版社，2020.  
- NVIDIA.《CUDA C++ Best Practices Guide》. NVIDIA Developer Documentation, 2024.

### 核心论文

| 作者 | 论文标题 | 发表信息 |
|------|----------|----------|
| Philippe Tillet, et al. | Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | MAPL, 2019 |

课程中还会发放额外的阅读材料和讲义，请关注课程主页和通知。
