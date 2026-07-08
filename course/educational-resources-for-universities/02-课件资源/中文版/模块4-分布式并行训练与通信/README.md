# 分布式并行训练与通信

课程子模块 IV（课时 28–36）。涵盖数据并行、张量并行、流水线并行、通信原语、拓扑优化、混合并行、容错与 Checkpointing，以及显存优化技术。

## 目录结构

```
06-Distributed-Parallel-Training-and-Communication/
├── 6-1-slides_zh/       # 中文讲义 + 作业 + 教学大纲
├── 6-2-slides_en/       # 英文讲义 + 作业 + 教学大纲
└── 6-3-videos_zh/       # 视频列表
```

## 教学大纲（Syllabus）

> 📄 中文版：`6-1-slides_zh/大纲_分布式并行训练与通信_Syllabus.docx`
> 📄 英文版：`6-2-slides_en/Distributed Parallel Training & Communication_Syllabus.docx`

建议在学习本模块前优先阅读大纲，了解课程目标、先修要求和学习路径。

### 课程目标

1. 掌握 DDP 与 Gradient Bucketing 技术
2. 理解模型并行（Tensor Parallel 数学推导、Pipeline Parallel 1F1B 调度）
3. 掌握集合通信原语与 Ring / Tree 优化
4. 使用 FlagScale 配置 3D 混合并行
5. 理解容错机制、异步 Checkpointing 与显存优化（Gradient Checkpointing、CPU Offload）
6. 使用 FlagScale + FlagCX 部署 7B 模型分布式训练

### 先修要求

- 完成模块 I–III（课时 01–27）或具备等效基础
- 计算机组成原理、操作系统原理、深度学习导论、C/C++ 编程

### 课程平台

- FlagOS 软件栈（FlagScale、FlagCX）
- 在线实验室：https://flagos.io/OnlineLab

### 模块期末项目

使用 FlagScale + FlagCX 在多 GPU 上部署 7B 模型分布式训练

### 参考资料

- Mohammad Shoeybi et al., *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism* (arXiv 2019)
- Samyam Rajbhandari et al., *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* (SC 2020)
- Deepak Narayanan et al., *Efficient Large-Scale Language Model Training on GPU Clusters* (SC 2021)

## 课时列表

| 课时 | 主题 |
|------|------|
| 28 | 数据并行 |
| 29 | 模型并行 I (Tensor Parallel) |
| 30 | 模型并行 II (Pipeline Parallel) |
| 31 | 通信原语 |
| 32 | 通信拓扑优化 |
| 33 | 混合并行 |
| 34 | 容错与 Checkpointing |
| 35 | 显存优化技术 |
| 36 | Module IV 期末项目 |

## 作业

| 编号 | 主题 |
|------|------|
| 9 | 并行策略基础 |
| 10 | 通信优化与显存管理 |

## 许可证

本课程内容采用 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/deed.zh) 协议进行许可，© 2026 FlagOS 社区。
