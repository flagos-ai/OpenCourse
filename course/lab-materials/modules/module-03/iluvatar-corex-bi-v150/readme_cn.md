# Triton Optimization Pass Lab（天数智芯 BI-V150）

[English](README.md) | 简体中文

使用天数 BI-V150 和平台预装的 CoreX、PyTorch、Triton，通过 blocked GEMM 与 fused softmax 观察 TTIR、TTGIR 和天数后端 LLIR 的编译优化。

仓库自带的 `artifacts/`、`lab-results/` 中原有的 IR、二进制和编译缓存是 H200 产物。

## 1. 验证环境

在实验平台进入仓库目录，直接使用预装的 `python3`：

```bash
python3 check_env.py
```

脚本检查实际 Python 和依赖路径、`ixsmi`、分配的 BI-V150 逻辑设备、Triton 后端和 warp 大小，并编译运行两个原有 kernel 做数值检查。检查覆盖非整块矩阵和非 2 的幂 softmax 列数；参考结果在 CPU 上使用 float64 计算。脚本不安装依赖、不修改平台环境配置，编译缓存使用临时目录。

- 检查通过返回 `0`；依赖、设备、编译或精度检查失败返回 `1`。
- `ixsmi` 未加入 PATH 时会提示警告，仍会继续检查 Python GPU 运行时。
- `--device-index` 是当前 Python 进程可见的逻辑编号，单卡容器通常用 `0`；不要根据 `/dev/iluvatarN` 的节点名推算。
- `torch.cuda` 和 `device="cuda"` 是 CoreX 使用的兼容接口，保留原有用法。
- 使用平台配套的依赖，不需要自行创建 Conda 环境或执行 `pip install -r requirements.txt`。

仅检查环境、不编译 kernel，或保存报告：

```bash
python3 check_env.py --skip-smoke
python3 check_env.py --device-index 0 --report environment.json
```

`--skip-smoke` 通过不代表 kernel 已验证。`requirements.txt` 中的版本约束不用于安装课程平台依赖；缺少库时按平台说明加载配套环境。

## 2. 在 V150 生成实验产物

以下命令在同一个 Bash 终端、仓库根目录执行。每次创建新目录，保留已有文件：

```bash
set -o pipefail
mkdir -p lab-results
V150_RUN_DIR="$(mktemp -d "$PWD/lab-results/v150.XXXXXX")"
printf '本次结果目录：%s\n' "$V150_RUN_DIR"
python3 check_env.py --device-index 0 --report "$V150_RUN_DIR/environment.json"
```

确认检查通过后，运行原有实验脚本：

```bash
python3 run_lab.py --kernel all --device-index 0 --dump-dir "$V150_RUN_DIR/baseline" 2>&1 | tee "$V150_RUN_DIR/baseline.log"
cat "$V150_RUN_DIR/baseline/matmul/run.json"
cat "$V150_RUN_DIR/baseline/softmax/run.json"
```

核对新记录中的 `device.name`、`compiled_metadata`、`artifact_stages` 和误差。实验脚本本身只记录误差；环境验证脚本额外提供独立的数值断言。

下文以 `artifacts/` 统称本次输出根目录，阅读时替换为 `$V150_RUN_DIR/baseline`。实际输出阶段由平台适配版编译器决定，常见目录为：

```text
baseline/
  matmul/
    00_ttir.mlir
    01_ttgir.mlir
    02_llir.ll
    04_cubin.bin
    run.json
  softmax/
    00_ttir.mlir
    01_ttgir.mlir
    02_llir.ll
    04_cubin.bin
    run.json
```

不要求后端导出 PTX。按 `Triton Python → TTIR → TTGIR → BI LLIR → 设备二进制` 阅读，具体文件名以 `run.json` 和实际目录为准。下面的天数 IR 片段用于讲解，布局、操作名和流水结构可能随 CoreX 版本及编译参数变化。

## 3. 从 TTIR 开始读：先看“算什么”

TTIR 与 Triton Python 最接近。此处仍能看到读内存、写内存、矩阵乘和求和等高层操作，还没有真正安排线程和共享临时区。

### 3.1 Softmax

先读本次输出目录中的 `artifacts/softmax/00_ttir.mlir`。一个 program 负责一行 softmax，主线是：

```text
找当前行
→ 读入 1024 个元素
→ 求这一行的最大值
→ 每个元素减最大值
→ 求指数
→ 求指数之和
→ 相除
→ 写回这一行
```

学生应回答：

1. 为什么读越界元素时填充负无穷？因为它不会错误地成为最大值。
2. 为什么整个过程不用把中间结果写回显存？因为读、求最大值、指数、求和、除法和写回都在同一个 kernel 内完成。

### 3.2 Matmul

再读本次输出目录中的 `artifacts/matmul/00_ttir.mlir`。找这五类操作：

| 在 TTIR 中找什么 | 它表示什么 |
|---|---|
| `tt.get_program_id` | 当前 program 负责 C 的哪一块 |
| `tt.load` | 读入当前需要的 A、B 小块 |
| `scf.for` | 沿 K 维分多轮累加 |
| `tt.dot` | 当前一轮的小矩阵乘 |
| `tt.store` | 把算完的 C 小块写回 |

部分天数适配版的 TTIR 会包含较多 i64 转换、范围比较和辅助函数，主要用于地址计算与安全检查。阅读时先追踪上面的五类操作。

## 4. TTGIR：看“谁做、放哪里、何时搬”

TTGIR 在 TTIR 的基础上加入了线程分工和片上临时存储。这里最重要的不是每个符号的拼写，而是回答：

```text
一块数据由哪些线程处理？
它从显存读出后放在哪里？
什么时候提前搬下一块？
最后怎样送进矩阵计算单元？
```

不同适配版本可能使用 `#triton_gpu.*` 或 `#ttg.*` 前缀，以本次输出为准。

### 4.1 `#triton_gpu.blocked`：普通数据的线程分工

matmul 的普通数据布局示例：

```mlir
#triton_gpu.blocked<{
  sizePerThread = [1, 8],
  threadsPerWarp = [16, 4],
  warpsPerCTA = [2, 2],
  ...
}>
```

可以这样理解：

| 字段 | 最简单的意思 |
|---|---|
| `sizePerThread=[1,8]` | 每个线程处理 1×8 个元素 |
| `threadsPerWarp=[16,4]` | 一个 64-thread warp 按 16×4 的方式分工 |
| `warpsPerCTA=[2,2]` | 四个 warp 按 2×2 的方式合作 |

它常用于读 A/B、生成地址、生成 mask 和普通的逐元素操作。可以把它看成一张“谁负责哪些元素”的分工表。

A 和 B 在内存中的方向不同，生成的 IR 可能为它们选择不同的 blocked layout。

### 4.2 `#triton_gpu.iluvatar_mma`：矩阵乘结果的分工

矩阵乘结果的布局示例：

```mlir
#triton_gpu.iluvatar_mma<{
  versionMajor = 1,
  warpsPerCTA = [2, 2],
  instrShape = [16, 16]
}>
```

它不是普通的线程分工，而是给矩阵计算单元准备的分工。含义是：C 的累加结果要按照 BI-V150 矩阵计算单元喜欢的方式分给四个 warp。

课堂上只需解释：

- `warpsPerCTA=[2,2]`：四个 warp 合作完成一块输出；
- `instrShape=[16,16]`：大矩阵乘会被拆成许多个 16×16 的小块；
- 不需要解释 `versionMajor` 的编号。

### 4.3 `#triton_gpu.shared`：A、B 在共享临时区中的摆放

共享临时区布局示例：

```mlir
#triton_gpu.shared<{..., useTcu = true}>
```

它说明 A、B 被搬到 CTA 内大家共用的临时区后，应该怎样摆放。`useTcu=true` 可以简单理解成：这个摆法是为 BI-V150 的矩阵计算单元准备的。

三阶段流水中可查找类似的临时区类型：

```mlir
!tt.memdesc<3x64x32xf16, #shared, mutable>
!tt.memdesc<3x32x64xf16, #shared1, mutable>
```

它们表示：

```text
三份 A 小块，每份 64×32
三份 B 小块，每份 32×64
```

第一个维度的 `3` 就是 `num_stages=3`。可以把它讲成三格抽屉：正在用一格计算时，其余抽屉可以提前装入后面要用的数据。

### 4.4 `dot_op`：送入矩阵乘前的最后整理

GEMM 的核心步骤是：

```text
从共享临时区取出 A
→ 整理为 A 的 dot_op
从共享临时区取出 B
→ 整理为 B 的 dot_op
tt.dot(A, B, 当前累加值)
→ 新的累加值
```

这里的 `#triton_gpu.dot_op` 表示：A 或 B 已经被整理成矩阵计算单元可以直接使用的形式。里面的 `useSme` 是本后端留下的额外标记，说明 A、B 的内部准备方式可能不同；不需要推测它的微架构细节。

`tt.dot` 仍然是最重要的锚点：高层矩阵乘没有消失，只是输入已经按硬件喜欢的方式准备好。

## 5. Software Pipelining：计算时准备下一块数据

K 维循环被分成多轮，每轮处理 `BLOCK_K=32`。如果每轮都先等待读完 A/B，再开始计算，矩阵计算单元会经常空等。

下面的三阶段流水示例采用这样的做法，需与本次实际 IR 对照：

```text
先准备三份 A/B 小块
→ 使用当前一份做矩阵乘
→ 同时搬运后面的 A/B 小块
→ 真正使用前再等待它准备完成
```

TTGIR 中的对应关系：

| 操作 | 最简单的解释 |
|---|---|
| `local_alloc` | 准备三格临时抽屉 |
| `async_copy_global_to_local` | 提前从显存搬 A/B |
| `async_commit_group` | 提交这一批搬运 |
| `async_wait` | 真正要用前确认数据已经到位 |
| `scf.for` | 主循环开始 |
| `local_load` 和 `tt.dot` | 取当前数据并计算 |
| 新一轮 copy 与 wait | 计算当前块时准备未来块 |
| 最后等待并释放临时区 | 循环结束时收尾 |

一句话讲解：

> 做当前一题时，先把后面两题的材料拿到桌上。这样当前一题做完后，不必再等材料送来。

`num_stages` 增大并不保证更快：抽屉更多会减少等待，也会占用更多片上空间和寄存器。

## 6. LLIR：确认真正使用了 BI-V150 的矩阵计算路径

天数目标的 `artifacts/matmul/02_llir.ll` 中可查找类似的目标标记：

```llvm
target triple = "bi-iluvatar-ilurt"
```

这说明它已经进入 BI-V150 的低层目标代码。此时不再重点讨论高层张量，而是观察四类低层操作：

| LLIR 中的线索 | 它表示什么 |
|---|---|
| `addrspace(1)` | 设备上的普通数据地址，例如 A、B、C |
| `addrspace(3) @global_smem` | CTA 内共享的临时区 |
| `llvm.bi.sme.load...` | 按该硬件喜欢的方式准备矩阵输入 |
| `llvm.bi.sl.waitcnt` | 等待先前的准备或搬运完成 |
| `llvm.bi.sl.barrier.alu` | 让合作线程在此处对齐 |
| `llvm.bi.matrix.mad.f32x4.f16x4` | fp16 输入相乘，并累加到 fp32；这是 `tt.dot` 的低层对应 |

在本次 matmul LLIR 中搜索这些操作，并按“准备输入 → 等待 → 同步 → 矩阵乘加”的顺序追踪。

不要逐行读 LLIR。只需让学生完成这条对应：

```text
TTIR 的 tt.dot
→ TTGIR 中准备好的 A/B、共享临时区和 dot_op
→ LLIR 中的 llvm.bi.matrix.mad
```

这说明高层的矩阵乘最终已经交给该设备的矩阵计算部件。

## 7. Softmax 的低层实现

softmax 没有矩阵乘。它的重点是多个线程一起求最大值和求和。

在 TTGIR 中，一个 CTA 有四个 64-thread warp；一个 warp 先处理自己负责的一部分元素，再把多个 warp 的部分结果合并。

在 LLIR 中：

| 线索 | 最简单的解释 |
|---|---|
| `llvm.bi.slb.shfl.idx.b32.i32` | 同一个 warp 内的线程互相交换局部结果 |
| `addrspace(3) @global_smem` | 不同 warp 把各自答案放到共同白板上 |
| `llvm.nvvm.barrier0()` | 所有 warp 到齐后再读取白板 |
| `llvm.exp2.f32` | 指数计算的低层实现 |

课堂可以这样说：

> 一行 1024 个数太多。先让每个小组找自己的最大值，再把四个小组的答案放到白板上，得到整行最大值。求和也做同样的事。整个过程没有把完整的中间数组写回显存，所以它仍然是融合的。

## 8. 学生任务

### 任务 1：改变 tile

改变 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 后，观察：

- `tt.dot` 的 A、B、C shape 是否变化；
- `iluvatar_mma` 的 `warpsPerCTA` 与 `instrShape` 是否变化；
- A/B 的三阶段临时区 shape 是否变化；
- `dot_op` 中的布局信息是否变化。

学生应解释：tile 变大可以增加一次搬运后可复用的数据，但也会占用更多临时区和寄存器。

### 任务 2：改变 num_warps

观察：

- module attribute 中的 `num-warps`；
- `#triton_gpu.blocked` 中的 `threadsPerWarp` 与 `warpsPerCTA`；
- `#triton_gpu.iluvatar_mma` 中的 `warpsPerCTA`；
- LLIR 中矩阵乘和临时数据准备的变化。

学生应先算出 CTA 线程数：

```text
CTA 线程数 = num_warps × 64
```

然后解释：更多 warp 能让更多线程分担一个 tile，但也会增加资源使用，因此不一定更快。

### 任务 3：改变 num_stages

观察：

- `local_alloc<STAGES x ...>` 的第一个维度；
- 主循环开始前预取了多少数据；
- `async_wait` 的位置和等待深度；
- benchmark 是否发生变化。

学生应解释：stage 数代表提前准备几批数据，而不是有几条独立执行流。

### 任务 4：比较 matmul 和 softmax

matmul 的重点是把 A/B 送进矩阵计算单元；softmax 的重点是融合和合作求最大值、求和。softmax 没有 `tt.dot`，因此不走 `llvm.bi.matrix.mad` 这条路径。

### 任务 5：从低层代码验证结果

对 matmul，找到 `llvm.bi.sme.load`、`llvm.bi.sl.waitcnt`、`llvm.bi.sl.barrier.alu` 和 `llvm.bi.matrix.mad`，解释它们分别对应“准备输入、等待、线程对齐、矩阵乘加”。

对 softmax，找到 shuffle、shared memory、barrier 和指数计算，解释它们分别对应“组内合并、组间合并、同步、指数”。

## 9. 主讲助教的讲解流程

### 第一段：先讲不变的东西

展示 Python kernel，告诉学生：矩阵乘和 softmax 的数学含义没有变。换 GPU 后，变化的是线程分工、临时存储和最后使用的专门计算部件。

### 第二段：TTIR，只追踪主线

矩阵乘只追踪：读 A/B、K 循环、`tt.dot`、写 C。

softmax 只追踪：读一行、最大值、指数、求和、写回。

如果学生被长长的地址检查干扰，提醒他们：那是编译器在核对地址，不是算法主线。

### 第三段：TTGIR，只讲三个问题

1. 谁做：一个 warp 有 64 人，四个 warp 按 2×2 合作。
2. 放哪：A/B 先进入共享临时区，准备了三格缓冲。
3. 何时搬：计算当前块时，提前搬后面的块。

讲解时可以把 shared memory 叫“公共临时桌”，把三阶段叫“三格抽屉”，把 `tt.dot` 叫“交给矩阵计算单元”。

### 第四段：LLIR，只找终点

不逐行阅读。只展示：

```text
准备输入：sme.load
等待准备完成：waitcnt
线程对齐：barrier
真正的矩阵乘：matrix.mad
```

然后把 `matrix.mad` 指回 TTIR 的 `tt.dot`，完成从 Python 到硬件相关操作的闭环。

### 第五段：用 softmax 收尾

强调 softmax 的优势是融合：中间结果不回显存。它需要线程合作求最大值和求和，但不需要矩阵计算单元。

最后让学生用一句话总结：

> Triton Python 说要算什么；TTGIR 安排谁来算、数据放哪；LLIR 把这个安排变成 BI-V150 真正执行的低层操作。

## 10. 容易讲错的点

- 不要把一个 warp 说成 32 个线程；本环境是 64 个线程。
- 不要把 `iluvatar_mma` 的编号解释成另一种设备的编号。
- 不要把 TTIR 中额外的 i64 检查当成算法优化。
- 不要把 `num_stages` 说成多条独立执行流；它是提前准备的数据缓冲深度。
- 不要逐行阅读 LLIR；只追踪输入准备、等待、同步和 `matrix.mad`。
- 应以本次 V150 运行的设备记录、BI LLIR 和最终设备二进制验证编译目标。

## 11. 生成不同配置的 V150 示例

先完成第 2 节的目录设置，再分别生成流水深度、warp 数和 tile 的对比结果：

```bash
for stages in 1 3 5; do
  python3 run_lab.py --kernel matmul --device-index 0 --num-stages "$stages" --dump-dir "$V150_RUN_DIR/stages_$stages"
done
for warps in 2 4 8; do
  python3 run_lab.py --kernel matmul --device-index 0 --num-warps "$warps" --dump-dir "$V150_RUN_DIR/warps_$warps"
done
python3 run_lab.py --kernel matmul --device-index 0 --block-m 32 --block-n 64 --block-k 32 --dump-dir "$V150_RUN_DIR/tile_32x64x32"
python3 run_lab.py --kernel matmul --device-index 0 --block-m 128 --block-n 64 --block-k 32 --dump-dir "$V150_RUN_DIR/tile_128x64x32"
```

某个配置若超出后端资源或支持范围，保留对应报错，先以默认配置完成实验。

单独计时与导出逐 pass 日志：

```bash
python3 run_lab.py --kernel matmul --device-index 0 --benchmark --dump-dir "$V150_RUN_DIR/benchmark"
MLIR_ENABLE_DUMP=1 python3 run_lab.py --kernel matmul --device-index 0 --dump-dir "$V150_RUN_DIR/mlir-dump" > "$V150_RUN_DIR/mlir-run.log" 2> "$V150_RUN_DIR/mlir-passes.log"
LLVM_IR_ENABLE_DUMP=1 python3 run_lab.py --kernel matmul --device-index 0 --dump-dir "$V150_RUN_DIR/llvm-dump" > "$V150_RUN_DIR/llvm-run.log" 2> "$V150_RUN_DIR/llvm-passes.log"
```

dump 开关以适配版 Triton 的支持为准。`run_mlir_step.py` 是独立的 pass 重放工具，使用内部 API；其部分接口与 Triton 3.1 有差异，不作为本实验运行的前置条件。

## 12. 可选：天数性能分析工具

导出 IR 不需要 profiler。需要运行时间线和 kernel 性能指标时，可分别使用 `ixsys` 与 `ixkn-cli`，用 `ixsmi` 查看设备状态。

```bash
ixsys --help
ixkn-cli --help
ixkn-cli --list-sets
ixsys -t cuda -o "$V150_RUN_DIR/matmul.ptrace" python3 run_lab.py --kernel matmul --device-index 0 --dump-dir "$V150_RUN_DIR/ixsys"
ixkn-cli --set default -o "$V150_RUN_DIR/matmul.ixkn" python3 run_lab.py --kernel matmul --device-index 0 --dump-dir "$V150_RUN_DIR/ixkn"
```

工具参数、Python 运行环境与虚拟化设备支持情况以当前平台版本为准。分别运行两个 profiler，在结果中定位 `matmul_kernel`，不要将初始化、随机数和参考计算的时间当作该 kernel 的执行时间；计时对比使用未开启 profiler 的 benchmark。

参考：[天数 Profiler 实验](https://gitee.com/yihan-long/day2-lab5)、[BI-V150 平台说明](https://moark.com/docs/compute/clusters_gpu/iluvatar/iluvatar_BI-V150_gpu)。
