# 现有实验列表

本文档汇总所有 FlagOS 课程的实验资源，按平台分类。

## 天数智芯 BI-V150

| 实验 | Gitee 地址 | 说明 |
|------|------------|------|
| Day 1 | https://gitee.com/zhao-zhiqiang2023/lab-day1-tianshu-v150 | GPU/NPU 编程基础、Triton 内核编程 |
| Day 2 | https://gitee.com/zhao-zhiqiang2023/lab-day2-tianshu-v150 | 性能调优、Buffer 模型分析、自动调优 |
| Day 3 | https://gitee.com/zhao-zhiqiang2023/lab-day3-tianshu-v150 | vLLM 大模型部署、FlagGems 集成 |

### Day 1 内容

- 环境检查与配置
- torch_npu/torch_cuda 入门
- Triton 向量加法实现
- Softmax 算子对比

### Day 2 内容

- 性能基线测试
- Buffer 模型分析
- Roofline 模型与自动调优
- 性能优化总结

### Day 3 内容

- vLLM 环境配置
- FlagGems 集成
- 大模型推理服务部署
- 客户端测试

## 华为 Ascend 910C

| 实验 | Gitee 地址 | 说明 |
|------|------------|------|
| Day 1 | https://gitee.com/jieran-zhang/lab_day1 | NPU 编程基础、Triton-Ascend 入门 |
| Day 2 | https://gitee.com/jieran-zhang/lab_day2 | 性能调优实践 |
| Day 3 | https://gitee.com/jieran-zhang/lab_day3 | 大模型部署 |

### 环境要求

- CANN 8.3.RC2
- torch_npu 2.8.0
- Triton-Ascend 3.2.0

## 版本兼容性

| 平台 | vLLM 版本 | Triton 版本 | 设备 API |
|------|-----------|-------------|----------|
| BI-V150 | 0.13.0+cu102 | OpenAI Triton | torch.cuda |
| Ascend 910C | 待确认 | Triton-Ascend | torch.npu |

## 实验对照表

| 课件模块 | 对应实验 | 说明 |
|-----------|----------|------|
| 模块 1 | Day 1 部分 | Triton 编程基础 |
| 模块 2 | Day 1-2 | 算子开发与优化 |
| 模块 4 | Day 3 | 分布式训练与推理 |

**注意**：实验与课件模块并非完全一一对应，教师可根据需要灵活调整。

## 获取帮助

如果在实验过程中遇到问题：

1. 查看实验仓库中的 README 文档
2. 访问 [FlagOS 开发者社区](https://flagos.io)
3. 在 [GitHub Issues](https://github.com/flagos-ai/docs/issues) 提问
