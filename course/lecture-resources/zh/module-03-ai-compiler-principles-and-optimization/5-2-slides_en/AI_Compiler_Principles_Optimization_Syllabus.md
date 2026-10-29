AI Compiler Principles & Optimization_Syllabus
Module III: AI Compiler Principles & Optimization
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
This sub-course systematically covers the core principles and engineering practice of AI compilers. Starting from the design motivation behind AI compilers, the course provides an in-depth analysis of intermediate representations (IR), MLIR architecture, operator fusion, loop optimization, memory planning, and other key techniques, with a particular focus on code generation and runtime integration for multi-chip backend support. Using FlagTree as the engineering practice vehicle, the course helps students understand the central role of compilers in the AI systems software stack.

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
Understand the design motivation and core value of AI compilers, and master the technical positioning of mainstream compilers such as XLA, TorchDynamo, and TVM.
Master the foundational concepts of Intermediate Representations (IR), including control flow graphs, data flow graphs, SSA form, and computation graph optimization principles.
Gain a deep understanding of the MLIR architecture, including the Dialect system, Pass design, and their application within FlagTree.
Master the implementation principles of common compiler optimizations such as operator fusion, loop-invariant code motion, and memory planning.
Understand the technical path for multi-chip backend support, from MLIR Lowering to target code (PTX, SPIR-V) and runtime interface unification.
Develop the practical ability to write a simple optimization Pass or extend hardware backend support for FlagTree.

Session Schedule

| Session | Topic & Content |
| --- | --- |
| 19 | AI Compiler Overview: XLA, TorchDynamo, TVM, and FlagTree positioning |
| 20 | Intermediate Representation Basics: Control flow graph, data flow graph, SSA & FX Graph export |
| 21 | MLIR Architecture Deep Dive: Dialect system & FlagTree extensions |
| 22 | Operator Fusion Pass: Graph matching algorithms, subgraph rewriting & Conv+BN fusion practice |
| 23 | Loop-Invariant Code Motion: Invariant identification & FlagTree loop optimization |
| 24 | Layout & Memory Planning: Tensor memory allocation, memory reuse, Dead Code Elimination |
| 25 | Multi-Chip Backend I: MLIR Lowering to PTX/SPIR-V target code |
| 26 | Multi-Chip Backend II: Runtime integration & hardware Driver interface unification |
| 27 | Module Final Project: Contributing an optimization Pass or extending operator support for FlagTree |

Textbooks and References
There is no required textbook for this course. Reading materials primarily consist of lecture notes, research papers, and online documentation. The following are recommended reference resources:

Recommended Reference Books
Chris Lattner, et al. MLIR: Scaling Compiler Infrastructure for Domain Specific Computation. CGO, 2021.
Alfred V. Aho, et al. Compilers: Principles, Techniques, and Tools (2nd Ed.). Pearson, 2006.

Key Papers

| Authors | Paper Title | Publication |
| --- | --- | --- |
| Tianqi Chen, et al. | TVM: An Automated End-to-End Optimizing Compiler for Deep Learning | OSDI, 2018 |
| Chris Lattner, et al. | MLIR: Scaling Compiler Infrastructure for Domain Specific Computation | CGO, 2021 |
| Jason Ansel, et al. | TorchDynamo: An Acquisition-Free JIT for PyTorch | arXiv, 2024 |

Additional reading materials and lecture notes will be distributed during the course. Please follow the course homepage and announcements.
