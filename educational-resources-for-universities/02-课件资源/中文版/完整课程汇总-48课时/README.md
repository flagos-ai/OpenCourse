# 智算系统软硬件基础（全量汇总）

本模块为**课程总目录**，包含全部 48 个课时的讲义、作业和视频索引。各子模块的独立材料请参见模块 03–07。

## 目录结构

```
02-AI-Computing-Hardware-and-Software-Fundamentals/
├── 2-1-slides_zh/       # 全部中文讲义（课时 01–48）+ 作业（1–13）+ 总教学大纲
├── 2-2-slides_en/       # 全部英文讲义（Lecture 01–48）+ 作业（1–13）+ 总教学大纲
└── 2-3-videos_zh/       # 全部视频索引
```

## 教学大纲（Syllabus）

> 📄 中文版：`2-1-slides_zh/大纲_智能计算系统软硬件基础_Syllabus.docx`
> 📄 英文版：`2-2-slides_en/Hardware and Software Foundations of Intelligent Computing Systems_Syllabus.docx`

这是**课程总大纲**，涵盖所有 5 个子模块的整体教学安排，建议在开始学习前优先阅读。

### 课程基本信息

- **课程编号**：04832240
- **学分/学时**：3 学分 / 48 学时（32 理论 + 16 实践）
- **开课学期**：2026 秋季
- **开课院系**：北京大学计算机学院
- **适用专业**：计算机科学、软件工程、人工智能
- **授课教师**：罗国杰

### 先修要求

- 计算机组成原理
- 操作系统原理
- 深度学习导论
- C/C++ 编程

> 不要求 CUDA / Triton / GPU 编程经验。

### 课程平台与实验环境

- **FlagOS 软件栈**：FlagScale、FlagGems、FlagTree、FlagCX
- **在线实验室**：https://flagos.io/OnlineLab

### 成绩评定

| 组成部分 | 占比 |
|----------|------|
| 实验作业（4 次，每次 10%） | 40% |
| 课程项目（设计 9% + 终期 21%） | 30% |
| 期末考试（闭卷） | 30% |

### 实验作业

| 实验 | 内容 |
|------|------|
| 实验 1 | 使用 Triton 实现 Softmax Kernel |
| 实验 2 | 算子融合（RMSNorm + Residual Connection，需达到基线 90% 性能） |
| 实验 3 | 编写 MLIR Pass（Conv+BN 融合） |
| 实验 4 | 分布式训练（多 GPU 部署 7B 模型训练） |

### 核心参考论文

- Philippe Tillet et al., *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* (MAPL 2019)
- Tri Dao et al., *FlashAttention*
- Tianqi Chen et al., *TVM* (OSDI 2018)
- Chris Lattner et al., *MLIR* (CGO 2021)
- Mohammad Shoeybi et al., *Megatron-LM* (arXiv 2019)

## 课程总览

| 子模块 | 课时 | 主题 |
|--------|------|------|
| I   | 01–09 | AI 系统软件基础与异构计算 |
| II  | 10–18 | 高性能 AI 算子与算子工程 |
| III | 19–27 | AI 编译器原理与优化 |
| IV  | 28–36 | 分布式并行训练与通信 |
| V   | 37–48 | 性能评测与下一代内核生成 |

## 作业列表

| 编号 | 主题 |
|------|------|
| 1 | 概念理解 |
| 2 | Triton 入门上机 |
| 3 | 内存优化 |
| 4 | 注意力机制与 Flash Attention |
| 5 | 算子融合与 Profiling |
| 6 | 算子库设计 |
| 7 | IR 与编译器 Pass |
| 8 | 后端代码生成 |
| 9 | 并行策略基础 |
| 10 | 通信优化与显存管理 |
| 11 | 性能分析方法论 |
| 12 | AI 辅助内核生成 |
| 13 | 全栈优化综合 |

## 许可证

本课程内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 协议进行许可，© 2026 FlagOS 社区。
