# AI 编译器原理与优化

课程子模块 III（课时 19–27）。涵盖 AI 编译器概述、中间表示（IR）、MLIR 架构、算符融合 Pass、循环优化、内存规划，以及多芯片后端代码生成与 Runtime 集成。

## 目录结构

```
05-AI-Compiler-Principles-and-Optimization/
├── 5-1-slides_zh/       # 中文讲义 + 作业 + 教学大纲
├── 5-2-slides_en/       # 英文讲义 + 作业 + 教学大纲
└── 5-3-videos_zh/       # 视频列表
```

## 教学大纲（Syllabus）

> 📄 中文版：`5-1-slides_zh/大纲_AI 编译器原理与优化_Syllabus.docx`
> 📄 英文版：`5-2-slides_en/AI Compiler Principles & Optimization_Syllabus.docx`

建议在学习本模块前优先阅读大纲，了解课程目标、先修要求和学习路径。

### 课程目标

1. 理解 AI 编译器的设计动机（XLA、TorchDynamo、TVM）
2. 掌握 IR 基础（控制流图、数据流图、SSA 形式）
3. 深入理解 MLIR 架构与 Dialect / Pass 设计
4. 掌握编译优化技术（算符融合、循环不变代码外提、内存规划）
5. 理解多芯片后端支持（MLIR Lowering → PTX / SPIR-V，Runtime 统一）
6. 为 FlagTree 编写优化 Pass

### 先修要求

- 完成模块 I–II（课时 01–18）或具备等效基础
- 计算机组成原理、操作系统原理、深度学习导论、C/C++ 编程

### 课程平台

- FlagOS 软件栈（FlagTree 编译器）
- 在线实验室：https://flagos.io/OnlineLab

### 模块期末项目

为 FlagTree 贡献一个优化 Pass（如 Conv+BN 融合）

### 参考资料

- Chris Lattner et al., *MLIR: Scaling Compiler Infrastructure for Domain Specific Computation* (CGO 2021)
- Alfred V. Aho et al.,《编译原理》（龙书，第 2 版）
- Tianqi Chen et al., *TVM: An Automated End-to-End Optimizing Compiler* (OSDI 2018)
- Jason Ansel et al., *TorchDynamo* (arXiv 2024)

## 课时列表

| 课时 | 主题 |
|------|------|
| 19 | AI 编译器概述 |
| 20 | 中间表示 (IR) 基础 |
| 21 | MLIR 架构深度解析 |
| 22 | 算符融合 Pass (Fusion Pass) |
| 23 | 循环不变代码外提 |
| 24 | 布局与内存规划 |
| 25 | 多芯片后端支持 I (Codegen) |
| 26 | 多芯片后端支持 II (Runtime 集成) |
| 27 | Module III 期末项目 |

## 作业

| 编号 | 主题 |
|------|------|
| 7 | IR 与编译器 Pass |
| 8 | 后端代码生成 |

## 许可证

本课程内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 协议进行许可，© 2026 FlagOS 社区。
