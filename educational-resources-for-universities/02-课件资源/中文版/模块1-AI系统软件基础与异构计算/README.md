# AI 系统软件基础与异构计算

课程子模块 I（课时 01–09）。涵盖 AI 硬件体系结构、异构计算编程模型、深度学习框架调用链路分析，以及 Triton 编程基础。

## 目录结构

```
03-AI-System-Software-and-Heterogeneous-Computing/
├── 3-1-slides_zh/       # 中文讲义 + 作业 + 教学大纲
├── 3-2-slides_en/       # 英文讲义 + 作业 + 教学大纲
└── 3-3-videos_zh/       # 视频列表
```

## 教学大纲（Syllabus）

> 📄 中文版：`3-1-slides_zh/大纲_AI 系统软件基础与异构计算_Syllabus.docx`
> 📄 英文版：`3-2-slides_en/AI Systems Software Foundations & Heterogeneous Computing_Syllabus.docx`

建议在学习本模块前优先阅读大纲，了解课程目标、先修要求和学习路径。

### 课程目标

1. 理解 CPU / GPU / NPU 的体系结构差异与适用场景
2. 掌握 SIMT 编程模型、线程与束（Warp）概念
3. 分析深度学习框架的调用链路（Python → C++ → CUDA Driver → Hardware）
4. 使用 Triton 语言编写基础 Kernel（Grid、Block、tl.load/tl.store）
5. 搭建 FlagOS 开发环境（Docker/Conda）
6. 理解内存层次结构优化（Shared Memory、Cache 局部性）

### 先修要求

- 计算机组成原理、操作系统原理、深度学习导论、C/C++ 编程
- 不要求 CUDA / Triton / GPU 编程经验

### 课程平台

- FlagOS 软件栈（FlagScale、FlagGems、FlagTree、FlagCX）
- 在线实验室：https://flagos.io/OnlineLab

### 模块期末项目

使用 Triton 实现 Softmax Kernel

### 参考资料

- 郑纬民等，《智能计算系统》（2020）
- NVIDIA CUDA Best Practices Guide（2024）
- Philippe Tillet et al., *Triton* (MAPL 2019)

## 课时列表

| 课时 | 主题 |
|------|------|
| 01 | 课程导论与 AI 硬件全景 |
| 02 | 异构计算编程模型基础 |
| 03 | 深度学习框架调用链路分析 |
| 04 | Triton 语言入门 |
| 05 | 环境搭建与工具链 |
| 06 | 向量计算基础 (Vector Add) |
| 07 | 矩阵乘法优化 I |
| 08 | 内存层次结构优化 |
| 09 | Module I 期中项目 |

## 作业

| 编号 | 主题 |
|------|------|
| 1 | 概念理解 |
| 2 | Triton 入门上机 |
| 3 | 内存优化 |

## 许可证

本课程内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 协议进行许可，© 2026 FlagOS 社区。
