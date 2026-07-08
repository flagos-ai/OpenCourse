# 性能评测与下一代内核生成

课程子模块 V（课时 37–48，共 12 学时）。涵盖性能分析指标、自动调优、算子 Profiling 深度实践、LLM 辅助编程、KernelGen 架构、异构系统优化、端到端性能优化、开源贡献实践，以及前沿技术展望。

## 目录结构

```
07-Performance-Benchmarking-and-Next-Gen-Kernel-Generation/
├── 7-1-slides_zh/       # 中文讲义 + 作业 + 教学大纲
├── 7-2-slides_en/       # 英文讲义 + 作业
└── 7-3-videos_zh/       # 视频列表
```

## 教学大纲（Syllabus）

> 📄 中文版：`7-1-slides_zh/大纲_性能评测与下一代内核生成_Syllabus.docx`

建议在学习本模块前优先阅读大纲，了解课程目标、先修要求和学习路径。

### 课程目标

1. 掌握性能指标体系（FLOPS、Bandwidth、Latency、Utilization）与 Roofline Model
2. 理解自动调优技术，使用 FlagPerf 进行基准测试
3. 进行深度 Profiling 分析（Occupancy、Memory Throughput、指令级分析）
4. 理解 LLM 辅助 KernelGen 的架构与 Prompt 工程
5. 掌握端到端性能优化方法
6. 学习开源贡献流程（代码规范、PR 提交、CI/CD）

### 先修要求

- 完成模块 I–IV（课时 01–36）或具备等效基础
- 计算机组成原理、操作系统原理、深度学习导论、C/C++ 编程

### 课程平台

- FlagOS 软件栈（FlagPerf、KernelGen）
- 在线实验室：https://flagos.io/OnlineLab

### 模块期末项目

课程综合项目答辩（个人或 2–3 人团队）

### 前沿展望

课时 48 将介绍前沿技术方向：WebGPU、MoE（Mixture of Experts）、FlagOS-Robo 等。

### 参考资料

- 郑纬民等，《智能计算系统》（2020）
- NVIDIA CUDA Best Practices Guide（2024）

## 课时列表

| 课时 | 主题 |
|------|------|
| 37 | 性能分析指标 |
| 38 | 自动调优 |
| 39 | 算子性能 Profiling 深度实践 |
| 40 | LLM 辅助编程概述 |
| 41 | KernelGen 的架构与 Prompt 工程 |
| 42 | 从生成到部署：生成代码的集成 |
| 43 | 异构系统的极致优化 |
| 44 | 端到端性能优化 |
| 45 | 开源贡献与社区协作 |
| 46-47 | 课程项目答辩 |
| 48 | 前沿技术展望与总结 |

## 作业

| 编号 | 主题 |
|------|------|
| 11 | 性能分析方法论 |
| 12 | AI 辅助内核生成 |
| 13 | 全栈优化综合 |

## 许可证

本课程内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 协议进行许可，© 2026 FlagOS 社区。
