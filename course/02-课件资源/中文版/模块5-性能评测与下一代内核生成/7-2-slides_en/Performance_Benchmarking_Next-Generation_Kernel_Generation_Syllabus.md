Performance Benchmarking & Next-Generation Kernel Generation
Module V: Performance Benchmarking & Next-Generation Kernel Generation
Syllabus

Course Information
Course Number: 04832240
Hours: 12 Hours
Semester: Fall 2026
Class Schedule: (See course registration system)
Classroom: (See course registration system)
Department: School of Computer Science, Peking University
Target Programs: Computer Science and Technology, Software Engineering, Artificial Intelligence
Course Type: Elective (Computer Architecture and Parallel Computing Track)

Instructor
Lead Instructor: Luo Guojie
Email: (TBD)
Office Hours: (TBD; appointments available via email)

Course Description
This sub-course serves as the capstone practice and frontier exploration module of the entire course. It systematically covers core performance analysis metrics and methodologies (Roofline Model, auto-tuning, deep profiling), explores the emerging trend of LLM-assisted automatic GPU Kernel generation (KernelGen), and connects the full course knowledge through end-to-end optimization case studies. The module also includes open-source contribution practice, course project presentations, and a forward-looking technology outlook, helping students integrate and apply everything learned throughout the course and develop the ability to independently conduct AI systems software research and optimization.

Course Platform and Online Laboratory Environment
FlagOS Intelligent Computing System Software Stack
The laboratory environment for this course is built on the FlagOS Intelligent Computing System Software Stack. Developed by the Beijing Academy of Artificial Intelligence (BAAI), FlagOS is a unified open-source heterogeneous computing software stack designed for diverse computing architectures, enabling a "develop once, run on multiple chips, deploy everywhere" paradigm. FlagOS comprises several core components, including:
FlagScale：A distributed training/inference framework supporting large model training and deployment across heterogeneous hardware
FlagGems：A general-purpose operator library consisting of high-performance AI operators implemented in the Triton language
FlagTree：A unified compiler supporting Triton compilation for multiple AI chip backends
FlagCX：A cross-chip communication library supporting high-performance collective communication in heterogeneous computing environments
All laboratory tasks, including Triton kernel development, operator optimization, MLIR pass implementation, and distributed training deployment, will be carried out within the FlagOS ecosystem.

Online Laboratory
The course provides a cloud-based online laboratory platform. Students can remotely access GPU computing resources through a web browser to complete all course experiments without configuring a local GPU environment.
Platform Registration and Login:
After the semester begins, the instructor will set up online laboratory accounts for all enrolled students. Please access the platform at the following URL:
Online Laboratory Portal: https://flagos.io/OnlineLab
Once your account is activated, you will be allocated GPU computing resources and can write, run, and debug code directly online. Please use computing time responsibly and avoid occupying resources for extended periods without releasing them.
Platform Usage Guide:
For detailed documentation on the online laboratory platform (environment setup, experiment submission procedures, FAQs, etc.), please refer to:
Platform Documentation: https://docs.flagos.io/projects/onlinelaboratory/en/latest/onlinelaboratory.html
Students are advised to carefully read the above documentation before the first experiment to familiarize themselves with the basic workflow. If you encounter technical issues, please consult the documentation first, then seek help through the course discussion forum or Office Hours.
Important Notes:
The computing resources on the online laboratory platform are exclusively for course experiments. Do not use them for personal projects unrelated to the course.
Please save your code and data locally in a timely manner after completing experiments, as platform instances may be periodically reclaimed.
If resources are unavailable during peak hours, please schedule your work during off-peak times and plan your experiment schedule accordingly.

Course-Related Links
Below are important links you may need during the course:

| Resource | URL |
| --- | --- |
| FlagOS Official Website | https://flagos.io |
| Online Laboratory | https://flagos.io/OnlineLab |
| Platform Documentation | https://docs.flagos.io/projects/onlinelaboratory/en/latest/onlinelaboratory.html |
| FlagOS Developer Community | https://flagos.io |
| FlagGems GitHub | https://github.com/FlagOpen/FlagGems |
| FlagTree GitHub | https://github.com/flagos-ai/FlagTree |
| FlagCX GitHub | https://github.com/flagos-ai/FlagCX |

Prerequisites
Before enrolling in this course, you should have the following background:
Computer Organization: Understanding of basic processor architecture, memory hierarchy, and instruction execution pipeline.
Operating Systems: Familiarity with fundamental concepts such as process/thread management and memory management.
Introduction to Deep Learning: Familiarity with basic neural network concepts and hands-on experience with PyTorch or TensorFlow.
C/C++ Programming: Solid proficiency in C/C++ programming.
If you have no prior experience with CUDA, Triton, or GPU programming, there is no need to worry—the course will start from the fundamentals. However, the course progresses at a brisk pace, so it is recommended that you familiarize yourself with basic Python and PyTorch usage in advance.

Course Objectives
Upon completing this course, you will be able to:
Master the core metrics framework for AI system performance analysis (FLOPS, Bandwidth, Latency, Utilization) and the Roofline Model analysis methodology.
Understand the basic principles of auto-tuning and be able to use FlagPerf for operator-level and model-level performance benchmarking.
Proficiently perform deep performance profiling, including Occupancy analysis, Memory Throughput analysis, and instruction-level analysis.
Understand the architecture and workflow of LLM-assisted code generation (KernelGen), and master Prompt engineering and generated code verification methods.
Develop end-to-end performance optimization capabilities, enabling coordinated optimization across the entire chain from algorithm design to compiler optimization to operator implementation.
Master the basic workflow of open-source contribution, including coding standards, PR submission, and CI/CD processes, and gain awareness of frontier developments in AI systems software.

Session Schedule

| Session | Topic & Content |
| --- | --- |
| 37 | Performance Analysis Metrics: FLOPS, Bandwidth, Latency, Roofline Model & FlagPerf |
| 38 | Auto-Tuning: Kernel Auto-tuning principles & FlagPerf performance comparison |
| 39 | Deep Profiling in Practice: Occupancy, Memory Throughput, instruction-level analysis |
| 40 | LLM-Assisted Programming Overview: Code generation technology & KernelGen project introduction |
| 41 | KernelGen Architecture & Prompt Engineering: LLM operator understanding & generated code verification |
| 42 | Generated Code Integration: Compilation, linking, runtime loading & FlagOS ecosystem integration |
| 43 | Extreme Optimization for Heterogeneous Systems: Micro-architecture specialization for specific hardware |
| 44 | End-to-End Performance Optimization: Algorithm → Compiler → Operator full-chain case study |
| 45 | Open-Source Contribution & Community Collaboration: Coding standards, PR submission & CI/CD workflow |
| 46 | Course Project Presentation I: Design proposal & preliminary progress |
| 47 | Course Project Presentation II: Final results & performance benchmarking data |
| 48 | Frontier Technology Outlook & Course Summary: WebGPU, MoE, FlagOS-Robo & emerging directions |

Textbooks and References
There is no required textbook for this course. Reading materials primarily consist of lecture notes, research papers, and online documentation. The following are recommended reference resources:

Recommended Reference Books
Zheng Weimin, Zhang Chenxi, et al. Intelligent Computing Systems. Tsinghua University Press, 2020.
NVIDIA. CUDA C++ Best Practices Guide. NVIDIA Developer Documentation, 2024.

Key Papers

| Authors | Paper Title | Publication |
| --- | --- | --- |
| Philippe Tillet, et al. | Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | MAPL, 2019 |
| Tianqi Chen, et al. | TVM: An Automated End-to-End Optimizing Compiler for Deep Learning | OSDI, 2018 |

Additional reading materials and lecture notes will be distributed during the course. Please follow the course homepage and announcements.
