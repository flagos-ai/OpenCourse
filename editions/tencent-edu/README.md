# Educational Resources for Tencent Education Platform

This directory contains comprehensive educational materials for the Tencent education platform edition.

## Overview

The program provides educational resources covering AI computing systems, from hardware foundations to software optimization.

## Directory Structure

```
editions/tencent-edu/
├── syllabus/                          # Course syllabus
├── general-education-course/          # AI Large Language Model General Course
├── textbooks/                         # Lecture slides (EN versions)
│   ├── en/                           # English version
│   │   ├── module-1-ai-systems-software-foundations-and-heterogeneous-computing/
│   │   ├── module-2-high-performance-ai-operators-and-operator-engineering/
│   │   ├── module-3-ai-compiler-principles-and-optimization/
│   │   ├── module-4-distributed-parallel-training-and-communication/
│   │   └── module-5-performance-analysis-methodology/
└── labs/                              # Hands-on labs
    # Day 1: NPU & Triton basics
    # Day 2: Performance tuning
    # Day 3: LLM deployment
```

## Course Modules

### General Education Course: AI Large Language Model

**Not part of the formal curriculum**, but serves as an engaging introduction to spark interest and guide learners into the subsequent modules.

| File | Description |
|------|-------------|
| `general-education-course/ai-large-language-model-general-course.pptx` | An accessible overview of large language models — what they are, how they work, and why they matter. Designed to motivate learners before diving into technical depth. |

**Purpose:**

- Bridge the gap for newcomers with limited AI background
- Introduce key concepts in an intuitive, non-technical manner
- Inspire curiosity and motivation to explore the full curriculum

### Module 1: AI Systems Software Foundations and Heterogeneous Computing

- Introduction to AI computing architecture
- Heterogeneous computing concepts
- NPU software stack

### Module 2: High-Performance AI Operators and Operator Engineering

- Triton kernel development
- Operator optimization techniques
- Vendor library comparison

### Module 3: AI Compiler Principles and Optimization

- Compiler fundamentals for AI
- Graph optimization
- Memory optimization

### Module 4: Distributed Parallel Training and Communication

- Data parallelism
- Model parallelism
- Communication optimization

### Module 5: Performance Analysis Methodology

- Profiling tools
- Roofline analysis
- Performance tuning workflow

## Hands-on Labs

| Day | Topic | Description |
|-----|-------|-------------|
| Day 1 | NPU & Triton | Introduction to software stack and Triton kernel development |
| Day 2 | Performance Tuning | Empirical modeling and autotuning techniques for optimal performance |
| Day 3 | LLM Deployment | Deploying and serving large language models locally |

Each lab directory contains its own README with detailed instructions.

## Languages

**Textbooks** are provided in two languages:

- **English (en):** `textbooks/en/` — Complete lecture materials in English

Other resources (syllabus, general education course, labs) are in their original language.

## License

CC BY-NC 4.0 — Creative Commons Attribution-NonCommercial 4.0 International.
