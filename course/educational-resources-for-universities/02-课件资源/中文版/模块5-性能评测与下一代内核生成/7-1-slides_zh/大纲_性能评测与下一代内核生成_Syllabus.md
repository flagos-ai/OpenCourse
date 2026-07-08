# 模块五：性能评测与下一代内核生成 (Module V: Performance Benchmarking & Next-Generation Kernel Generation: Syllabus)

## 课程基本信息

- **课程编号**：04832240  
- **学时**：12 学时  
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

本子课程为课程的综合实践与前沿探索模块。课程系统讲授性能分析的核心指标与方法论（Roofline Model、自动调优、深度 Profiling），探索大语言模型辅助 GPU Kernel 自动生成（KernelGen）的新趋势，并通过端到端优化案例串联全课程知识。课程同时包含开源贡献实践、课程项目答辩以及前沿技术展望，帮助同学们将全课程所学融会贯通，具备独立开展 AI 系统软件研发与优化的能力。

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

- 掌握 AI 系统性能分析的核心指标体系（FLOPS、Bandwidth、Latency、Utilization）及 Roofline Model 分析方法。  
- 理解自动调优（Auto-tuning）的基本原理，能够使用 FlagPerf 进行算子和模型级别的性能评测。  
- 熟练进行深度性能 Profiling，包括 Occupancy 分析、Memory Throughput 分析和指令级分析。  
- 理解 LLM 辅助代码生成（KernelGen）的架构与工作流，掌握 Prompt 工程与生成代码验证方法。  
- 具备端到端性能优化能力，能够从算法设计、编译器优化到算子实现进行全链路协同优化。  
- 掌握开源贡献的基本流程，包括代码规范、PR 提交与 CI/CD 流程，并了解 AI 系统软件的前沿发展方向。

## 课时安排

| 课时 | 主题与内容 |
|------|------------|
| 37 | 性能分析指标：FLOPS、Bandwidth、Latency、Roofline Model 与 FlagPerf |
| 38 | 自动调优：Kernel Auto-tuning 原理与 FlagPerf 性能对比 |
| 39 | 深度 Profiling 实践：Occupancy、Memory Throughput、指令级分析 |
| 40 | LLM 辅助编程概述：代码生成技术与 KernelGen 项目介绍 |
| 41 | KernelGen 架构与 Prompt 工程：LLM 理解算子逻辑与生成代码验证 |
| 42 | 生成代码的集成：编译、链接、运行时加载与 FlagOS 生态集成 |
| 43 | 异构系统极致优化：针对特定架构的微架构特化优化 |
| 44 | 端到端性能优化：算法→编译器→算子全链路协同优化案例 |
| 45 | 开源贡献与社区协作：代码规范、PR 提交与 CI/CD 流程 |
| 46 | 课程项目答辩 I：设计方案与初步进展展示 |
| 47 | 课程项目答辩 II：最终成果与性能测试数据展示 |
| 48 | 前沿技术展望与课程总结：WebGPU、MoE、FlagOS-Robo 等新兴方向 |

## 教材与参考资料

本课程没有指定的必修教材。阅读材料以课堂讲义、论文和在线文档为主。以下为推荐的参考资源：

### 推荐参考书

- 郑纬民，张晨曦，等.《智能计算系统》. 清华大学出版社，2020.  
- NVIDIA.《CUDA C++ Best Practices Guide》. NVIDIA Developer Documentation, 2024.

### 核心论文

| 作者 | 论文标题 | 发表信息 |
|------|----------|----------|
| Philippe Tillet, et al. | Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | MAPL, 2019 |
| Tianqi Chen, et al. | TVM: An Automated End-to-End Optimizing Compiler for Deep Learning | OSDI, 2018 |

课程中还会发放额外的阅读材料和讲义，请关注课程主页和通知。
