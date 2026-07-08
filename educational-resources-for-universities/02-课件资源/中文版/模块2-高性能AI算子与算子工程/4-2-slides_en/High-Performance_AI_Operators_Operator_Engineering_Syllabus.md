High-Performance AI Operators & Operator Engineering_Syllabus
Module II: High-Performance AI Operators & Operator Engineering
Syllabus

Course Information
Course Number: 04832240
Hours: 9 Hours
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
This sub-course delves into the high-performance implementation techniques of core operators in deep learning. Centered on key operators from the Transformer architecture—including the attention mechanism, convolution, normalization, and activation functions—the course systematically covers IO-aware algorithms such as Flash Attention, operator fusion strategies, and performance profiling methods. Using the FlagGems operator library and FlagAttention as engineering practice vehicles, the course helps students master the complete workflow of operator development from algorithm design to production-grade implementation.

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
Understand the computational principles of the attention mechanism in the Transformer architecture, and master the IO-aware design philosophy and Tiled Memory Access implementation strategy of Flash Attention.
Master multiple technical approaches to convolution operator optimization (Im2Col+GEMM, Direct Convolution, Winograd) and their applicable scenarios.
Be able to implement efficient normalization (LayerNorm, RMSNorm) and activation function (GELU) operators, and understand the design trade-offs between atomic operations and fused kernels.
Master operator fusion techniques: fuse multiple element-wise operations into a single Triton Kernel to reduce Kernel Launch overhead.
Proficiently use Nsight Compute and Nsight Systems for warp-level performance analysis and be able to identify operator performance bottlenecks.
Understand the architectural design principles of operator libraries, including dispatch mechanisms, interface abstraction, and multi-hardware adaptation strategies.

Session Schedule

| Session | Topic & Content |
| --- | --- |
| 10 | Attention Mechanism Principles: Q/K/V computation, Flash Attention concept, FlagAttention design goals |
| 11 | Flash Attention Implementation I: Tiled Memory Access, reducing HBM accesses |
| 12 | Flash Attention Implementation II: Masking handling & multi-architecture adaptation |
| 13 | Convolution Operator Optimization: Im2Col+GEMM, Direct Convolution, Winograd |
| 14 | Normalization & Activation Functions: Efficient LayerNorm/GELU implementation |
| 15 | Operator Fusion: Bias+Add+ReLU fusion in practice |
| 16 | Operator Debugging & Profiling: Nsight Compute/Systems warp-level analysis |
| 17 | Operator Library Architecture: Dispatch mechanism & FlagGems scheduler |
| 18 | Module Final Project: Implementing a fused RMSNorm + Residual Connection operator |

Textbooks and References
There is no required textbook for this course. Reading materials primarily consist of lecture notes, research papers, and online documentation. The following are recommended reference resources:

Recommended Reference Books
NVIDIA. CUDA C++ Best Practices Guide. NVIDIA Developer Documentation, 2024.
NVIDIA. Nsight Compute Documentation. NVIDIA Developer, 2024.

Key Papers

| Authors | Paper Title | Publication |
| --- | --- | --- |
| Tri Dao, et al. | FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | NeurIPS, 2022 |
| Tri Dao | FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | ICLR, 2024 |
| Philippe Tillet, et al. | Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations | MAPL, 2019 |

Additional reading materials and lecture notes will be distributed during the course. Please follow the course homepage and announcements.
