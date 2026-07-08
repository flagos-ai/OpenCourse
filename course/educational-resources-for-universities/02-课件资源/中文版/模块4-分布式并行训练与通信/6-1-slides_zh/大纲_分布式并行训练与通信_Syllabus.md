# 模块四：分布式并行训练与通信 (Module IV: Distributed Parallel Training & Communication: Syllabus)

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

本子课程系统讲授大规模分布式并行训练的核心技术与工程实践。课程涵盖数据并行、张量并行、流水线并行等主流并行范式，深入解析 AllReduce、AllGather 等通信原语及拓扑优化策略，并探讨容错、Checkpointing 和显存优化等关键工程问题。课程以 FlagScale 和 FlagCX 为工程实践平台，帮助同学们掌握在多机多卡环境下部署大模型训练任务的完整能力。

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

- 掌握数据并行（DDP）的工作原理，理解 Gradient Bucketing 等优化策略。  
- 理解模型并行的两种主要范式：Tensor Parallel（列并行与行并行的数学推导）和 Pipeline Parallel（1F1B 调度、Bubble 优化）。  
- 掌握主流集合通信原语（AllReduce、AllGather、Broadcast）及 Ring/Tree 拓扑优化算法。  
- 理解 3D 混合并行的设计原理，具备在 FlagScale 中配置多种并行策略的实践能力。  
- 掌握分布式训练中的容错机制、异步 Checkpointing 以及显存优化技术（Gradient Checkpointing、CPU Offload）。  
- 能够使用 FlagScale 和 FlagCX 在多机环境下完成 7B 参数量级别模型的训练部署。

## 课时安排

| 课时 | 主题与内容 |
|------|------------|
| 28 | 数据并行：DDP 原理、Gradient Bucketing、FlagScale LLM 微调 |
| 29 | 模型并行 I：Megatron-LM 架构，Column/Row Parallel 数学推导 |
| 30 | 模型并行 II：Pipeline Parallel，1F1B 策略与微批次调度 |
| 31 | 通信原语：AllReduce、AllGather、Broadcast 与 FlagCX 抽象层 |
| 32 | 通信拓扑优化：Ring/Tree-AllReduce、NVLink/InfiniBand 带宽利用 |
| 33 | 混合并行：3D 并行策略与 FlagScale 自动模型划分 |
| 34 | 容错与 Checkpointing：故障处理、异步保存、节点恢复模拟 |
| 35 | 显存优化技术：Gradient Checkpointing、CPU Offload 与 FlagScale 工具 |
| 36 | 模块期末项目：多机部署 7B 模型训练任务 |

## 教材与参考资料

本课程没有指定的必修教材。阅读材料以课堂讲义、论文和在线文档为主。以下为推荐的参考资源：

### 推荐参考书

- Shoeybi, et al.《Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism》. arXiv, 2019.  
- 郑纬民，张晨曦，等.《智能计算系统》. 清华大学出版社，2020.

### 核心论文

| 作者 | 论文标题 | 发表信息 |
|------|----------|----------|
| Shoeybi, et al. | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | arXiv, 2019 |
| Rajbhandari, et al. | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | SC, 2020 |
| Narayanan, et al. | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | SC, 2021 |

课程中还会发放额外的阅读材料和讲义，请关注课程主页和通知。
