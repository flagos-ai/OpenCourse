# 高性能 AI 算子与算子工程

课程子模块 II（课时 10–18）。涵盖注意力机制、Flash Attention、卷积优化、归一化与激活函数、算子融合、Profiling 调试，以及算子库架构设计。

## 目录结构

```
04-High-Performance-AI-Operators-and-Engineering/
├── 4-1-slides_zh/       # 中文讲义 + 作业 + 教学大纲
├── 4-2-slides_en/       # 英文讲义 + 作业 + 教学大纲
└── 4-3-videos_zh/       # 视频列表
```

## 教学大纲（Syllabus）

> 📄 中文版：`4-1-slides_zh/大纲_高性能 AI 算子与算子工程_Syllabus.docx`
> 📄 英文版：`4-2-slides_en/High-Performance AI Operators & Operator Engineering_Syllabus.docx`

建议在学习本模块前优先阅读大纲，了解课程目标、先修要求和学习路径。

### 课程目标

1. 理解 Flash Attention 的 IO 感知算法设计原理
2. 掌握卷积算子优化策略（Im2Col+GEMM、Direct Conv、Winograd）
3. 实现归一化与激活函数算子
4. 掌握算子融合技术与优化策略
5. 使用 Nsight Compute/Systems 进行性能 Profiling
6. 理解算子库架构设计（Dispatch 机制）

### 先修要求

- 完成模块 I（课时 01–09）或具备等效基础
- 计算机组成原理、操作系统原理、深度学习导论、C/C++ 编程

### 课程平台

- FlagOS 软件栈（FlagGems、FlagAttention）
- 在线实验室：https://flagos.io/OnlineLab

### 模块期末项目

实现 RMSNorm + Residual Connection 算子融合（需达到基线 90% 性能）

### 参考资料

- 郑纬民等，《智能计算系统》（2020）
- NVIDIA CUDA Best Practices Guide（2024）
- Philippe Tillet et al., *Triton* (MAPL 2019)

## 课时列表

| 课时 | 主题 |
|------|------|
| 10 | 注意力机制原理与算子需求 |
| 11 | Flash Attention 实现 I（IO 感知） |
| 12 | Flash Attention 实现 II（算法优化） |
| 13 | 卷积算子优化 (Convolution) |
| 14 | 归一化与激活函数 |
| 15 | 算子融合 |
| 16 | 复杂算子调试与 Profiling |
| 17 | 算子库的架构设计 |
| 18 | Module II 期末项目 |

## 作业

| 编号 | 主题 |
|------|------|
| 4 | 注意力机制与 Flash Attention |
| 5 | 算子融合与 Profiling |
| 6 | 算子库设计 |

## 许可证

本课程内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 协议进行许可，© 2026 FlagOS 社区。
