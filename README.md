[<img width="2182" height="602" alt="FlagOS banner" src="https://github.com/flagos-ai/FlagCX/blob/main/.github/assets/banner-20260130.png" />](https://flagos.io/)

<div align="right">
  <a href="https://www.linkedin.com/company/flagos-community" target="_blank">
    <img src="https://github.com/flagos-ai/OpenCourse/blob/main/.github/assets/Linkedin.png" alt="LinkedIn" width="32" height="32" />
  </a>

  <a href="https://www.youtube.com/@FlagOS_Official" target="_blank">
    <img src="https://github.com/flagos-ai/OpenCourse/blob/main/.github/assets/youtube.png" alt="YouTube" width="32" height="32" />
  </a>

  <a href="https://x.com/FlagOS_Official" target="_blank">
    <img src="https://github.com/flagos-ai/OpenCourse/blob/main/.github/assets/x.png" alt="X" width="32" height="32" />
  </a>

  <a href="https://www.facebook.com/flagosglobalcommunity/" target="_blank">
    <img src="https://github.com/flagos-ai/OpenCourse/blob/main/.github/assets/Facebook.png" alt="Facebook" width="32" height="32" />
  </a>

  <a href="https://discord.com/invite/ubqGuFMTNE" target="_blank">
    <img src="https://github.com/flagos-ai/OpenCourse/blob/main/.github/assets/discord.png" alt="Discord" width="32" height="32" />
  </a>
</div>

# FlagOS OpenCourse

Open educational resources for the [FlagOS](https://flagos.io/) open-source AI system software stack — lectures, hands-on labs, tutorials, and teaching guides covering the full stack from AI computing hardware fundamentals to compilers, high-performance operators, and distributed training.

> 🎓 The online learning center will be available at [https://edu.flagos.io](https://edu.flagos.io)

## Repository Structure

| Directory | Description |
|-----------|-------------|
| [`educational-resources-for-universities/`](educational-resources-for-universities/) | 高校合作支持中心（中文）：48 课时模块化课程课件、实验材料包、课程大纲示例、教师培训指南、在线实验室与高校合作流程 |
| [`educational-resources-for-platform/`](educational-resources-for-platform/) | Platform program (English): syllabus, general education course, five textbook modules, and three days of hands-on labs |
| [`educational-resources-for-china-ai-compute-faculty-development-program/`](educational-resources-for-china-ai-compute-faculty-development-program/) | Faculty development program: syllabus, general education course, textbook modules in English and French, and hands-on labs |
| [`resources/`](resources/) | Shared resources: hands-on tutorials for FlagOS components and course teaching syllabus |

## University Course (中文课程体系)

A comprehensive 48-lecture modular course for universities, covering the full AI system software stack:

| Module | Topic |
|--------|-------|
| 1 | AI 系统软件基础与异构计算 / AI Systems Software Foundations & Heterogeneous Computing |
| 2 | 高性能 AI 算子与算子工程 / High-Performance AI Operators & Operator Engineering |
| 3 | AI 编译器原理与优化 / AI Compiler Principles & Optimization |
| 4 | 分布式并行训练与通信 / Distributed Parallel Training & Communication |
| 5 | 性能评测与下一代内核生成 / Performance Benchmarking & Next-Gen Kernel Generation |

Chinese and English slides, homework, and lecture video indexes are provided under [`educational-resources-for-universities/02-课件资源/`](educational-resources-for-universities/02-课件资源/). Universities can access online labs with GPU/NPU compute support — no local hardware required. See [`08-高校合作流程/`](educational-resources-for-universities/08-高校合作流程/) for how to participate.

## Hands-on Tutorials

Step-by-step tutorials for each core FlagOS component, under [`resources/tutorial/`](resources/tutorial/):

| Component | Description |
|-----------|-------------|
| [**FlagGems**](https://github.com/flagos-ai/FlagGems) | High-performance general-purpose AI operator library (Triton-based) |
| [**FlagTree**](https://github.com/flagos-ai/FlagTree) | Unified AI compiler for multi-chip backends |
| [**FlagScale**](https://github.com/flagos-ai/FlagScale) | Large-scale distributed training and inference framework |
| [**FlagCX**](https://github.com/flagos-ai/FlagCX) | Unified cross-chip communication library |

## Labs

Hands-on lab materials (NPU & Triton basics, performance tuning, and LLM deployment) are available under the `labs/` directory of each program, with environment check scripts and reference implementations.

## Contributing & Contact

- Issues and suggestions: [GitHub Issues](https://github.com/flagos-ai/OpenCourse/issues)
- FlagOS community: [https://flagos.io](https://flagos.io) · [Discord](https://discord.com/invite/ubqGuFMTNE)

## License

Course materials in this repository are licensed under [CC BY-SA 4.0](./LICENSE) unless otherwise noted in a subdirectory's own `LICENSE` file.
