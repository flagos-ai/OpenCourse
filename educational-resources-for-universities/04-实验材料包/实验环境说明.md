# 实验环境说明

本文档说明 FlagOS 课程实验的环境配置和使用方式。

## 推荐方式：使用在线实验室

FlagOS 提供**在线实验室**，学生无需本地配置环境，直接通过浏览器访问 GPU/NPU 计算资源。

### 优势

- **零配置**：无需安装驱动、CUDA、PyTorch 等
- **即开即用**：登录即可开始实验
- **资源保障**：提供稳定的计算资源配额
- **多平台支持**：支持天数智芯、华为 Ascend 等多种芯片

### 使用步骤

1. 访问 [FlagOS 在线实验室](https://flagos.io/OnlineLab)
2. 使用分配的账号登录
3. 选择实验环境和计算资源
4. 开始实验

详细使用说明请参考 [在线实验室指南](../07-在线实验室指南/)。

## 可选方式：本地环境搭建

如高校有本地 GPU/NPU 资源，可选择本地搭建实验环境。

### 天数智芯 BI-V150 环境要求

```
# 软件版本
CoreX SDK >= 4.4.0
Python 3.12
PyTorch >= 2.0
Triton (OpenAI Triton)

# 硬件要求
天数智芯 BI-V150 GPU
```

### 华为 Ascend 910C 环境要求

```
# 软件版本
CANN 8.3.RC2
torch_npu 2.8.0
Triton-Ascend 3.2.0
Python 3.10+

# 硬件要求
华为 Ascend 910C NPU
```

### 环境检查

运行以下命令检查环境是否正确配置：

```python
# 天数智芯
import torch
print(torch.cuda.is_available())  # 应返回 True

# 华为 Ascend
import torch
import torch_npu
print(torch.npu.is_available())  # 应返回 True
```

## Docker 环境（可选）

我们提供 Docker 镜像，方便快速部署实验环境。

```bash
# 拉取镜像（示例，具体镜像地址请联系我们）
docker pull flagos/lab-env:latest

# 启动容器
docker run -it --gpus all flagos/lab-env:latest
```

## 常见问题

### 1. vLLM 版本不兼容

确保 vLLM 版本 >= 0.13.0，否则 vllm-plugin-FL 无法使用。

### 2. Triton 安装失败

- 天数智芯：`pip install triton`
- 华为 Ascend：`pip install triton-ascend`

### 3. 首次启动超时

设置环境变量：
```bash
export VLLM_ENGINE_ITERATION_TIMEOUT_S=36000
```

### 4. pip 安装慢

使用清华镜像：
```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple [package]
```

## 技术支持

如遇到环境问题，请联系：

- GitHub Issues：https://github.com/flagos-ai/docs/issues
- FlagOS 开发者社区：https://flagos.io
