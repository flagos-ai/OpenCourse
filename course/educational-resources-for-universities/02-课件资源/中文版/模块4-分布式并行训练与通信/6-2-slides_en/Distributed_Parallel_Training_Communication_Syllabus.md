Distributed Parallel Training & Communication_Syllabus
Module IV: Distributed Parallel Training & Communication
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
This sub-course systematically covers the core techniques and engineering practice of large-scale distributed parallel training. The course addresses major parallel paradigms including data parallelism, tensor parallelism, and pipeline parallelism; provides an in-depth analysis of communication primitives such as AllReduce and AllGather along with topology optimization strategies; and discusses key engineering challenges including fault tolerance, checkpointing, and memory optimization. Using FlagScale and FlagCX as engineering practice platforms, the course equips students with the complete ability to deploy large model training tasks in multi-node, multi-GPU environments.

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
Master the working principles of Data Parallel (DDP), and understand optimization strategies such as Gradient Bucketing.
Understand the two main paradigms of model parallelism: Tensor Parallel (mathematical derivation of Column Parallel and Row Parallel) and Pipeline Parallel (1F1B scheduling, Bubble optimization).
Master mainstream collective communication primitives (AllReduce, AllGather, Broadcast) and Ring/Tree topology optimization algorithms.
Understand the design principles of 3D hybrid parallelism and develop the practical ability to configure multiple parallel strategies in FlagScale.
Master fault tolerance mechanisms, asynchronous checkpointing, and memory optimization techniques (Gradient Checkpointing, CPU Offload) in distributed training.
Be able to use FlagScale and FlagCX to deploy a training task for a model with 7B parameters in a multi-node environment.

Session Schedule

| Session | Topic & Content |
| --- | --- |
| 28 | Data Parallelism: DDP principles, Gradient Bucketing, FlagScale LLM fine-tuning |
| 29 | Model Parallelism I: Megatron-LM architecture, Column/Row Parallel mathematical derivation |
| 30 | Model Parallelism II: Pipeline Parallel, 1F1B strategy & micro-batch scheduling |
| 31 | Communication Primitives: AllReduce, AllGather, Broadcast & FlagCX abstraction layer |
| 32 | Communication Topology Optimization: Ring/Tree-AllReduce, NVLink/InfiniBand bandwidth utilization |
| 33 | Hybrid Parallelism: 3D parallel strategies & FlagScale automatic model partitioning |
| 34 | Fault Tolerance & Checkpointing: Failure handling, async saving, node recovery simulation |
| 35 | Memory Optimization: Gradient Checkpointing, CPU Offload & FlagScale tools |
| 36 | Module Final Project: Multi-node deployment of a 7B model training task |

Textbooks and References
There is no required textbook for this course. Reading materials primarily consist of lecture notes, research papers, and online documentation. The following are recommended reference resources:

Recommended Reference Books
Shoeybi, et al. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. arXiv, 2019.
Zheng Weimin, Zhang Chenxi, et al. Intelligent Computing Systems. Tsinghua University Press, 2020.

Key Papers

| Authors | Paper Title | Publication |
| --- | --- | --- |
| Shoeybi, et al. | Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism | arXiv, 2019 |
| Rajbhandari, et al. | ZeRO: Memory Optimizations Toward Training Trillion Parameter Models | SC, 2020 |
| Narayanan, et al. | Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM | SC, 2021 |

Additional reading materials and lecture notes will be distributed during the course. Please follow the course homepage and announcements.
