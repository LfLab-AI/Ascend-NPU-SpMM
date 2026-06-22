# 昇腾 910B SpMM 算子实现说明与测试总结

> 适用版本：多 tiling 特化 kernel + host 运行时 dispatch 版本。  
> 主要文件：`spmm_kernel.cce`、`spmm.h`、`main.cpp`、`makefile`、`run.sh`、`test.sh`、`prof.py`、`spmm_gen_data.py`、`generate_full_datasets.py`、`use_full_dataset.py`。  
> 当前块大小：`d = 16`。A 矩阵按 16 行为一个向量块组织，因此当前数据生成要求 `M % 16 == 0`。N、K 不要求为 16 的倍数，host 和 kernel 已支持对应尾块处理。

---

## 1. 算子实现的功能

本算子实现结构化稀疏矩阵与稠密矩阵乘法：

```text
C = A_sparse × B_dense
```

矩阵含义如下：

```text
A: M × K，结构化稀疏矩阵，使用 vector-CSR 格式
B: K × N，稠密矩阵，输入为 row-major
C: M × N，输出矩阵，kernel 当前写回为 column-major
```

A 的稀疏结构不是普通逐元素 CSR，而是以 **16 行竖直向量块**为非零基本单元：一个非零块位于同一列、连续 16 行，包含 16 个数值。这样，一个 16 行的 A 子块可以与 B 中若干行组成一个小矩阵乘法，从而交给昇腾 Cube 单元计算。

当前实现目标：

1. 支持不同规模、不同稀疏度的 SpMM 测试；
2. 支持多组 tiling kernel 一次编译，host 运行时自动选择；
3. 支持 N、K 非规则值，其中 N 由 host 侧 padding 到当前计算批次需要的对齐粒度；
4. 在 910B 上针对中大规模、中高稀疏度场景获得较高吞吐。

---

## 2. 算子的数学逻辑

### 2.1 标准矩阵乘法

对于：

```text
A ∈ R^(M × K)
B ∈ R^(K × N)
C ∈ R^(M × N)
```

有：

```text
C[i, j] = Σ A[i, k] × B[k, j]
```

由于 A 是稀疏矩阵，大部分 `A[i, k]` 为 0，实际只需要对 A 的非零向量块参与计算。

### 2.2 vector-CSR 的计算视角

A 被按 16 行划分为多个 row block。第 `r` 个 row block 覆盖：

```text
A 的第 16r 行 ~ 第 16r+15 行
```

在某个 row block 中，每个非零向量块包含：

```text
1 个列号：说明它位于 A 的哪一列，也说明需要读取 B 的哪一行
16 个数值：对应这个列号下连续 16 行的 A 值
```

对于一个 row block，可以把其中所有非零向量块拼成一个小矩阵：

```text
A_block:     16 × 非零向量块数
B_selected:  非零向量块数 × N
C_block:     16 × N

C_block = A_block × B_selected
```

这里的 `B_selected` 不是 B 的连续前几行，而是根据 A 的非零列号从 B 中挑出来的若干行。

### 2.3 Cube 实际计算形式

为了适配 Cube 矩阵乘接口，kernel 实际计算的是上式的转置形式：

```text
C_block^T = B_selected^T × A_block^T
```

对应到硬件数据流，可以理解为：

```text
左矩阵：B_selected^T，形状约为 N批次 × 稀疏K批次
右矩阵：A_block^T，形状约为 稀疏K批次 × 16
输出：  C_block^T，形状约为 N批次 × 16
```

最终输出仍然表示原始的 `C_block = A_block × B_selected`，只是为了让 Cube 高效执行，内部采用转置后的矩阵乘法组织。

---

## 3. 输入输出格式

### 3.1 A：vector-CSR 格式

A 使用 `vector_csr/` 目录下的三类文件：

| 文件 | 类型 | 含义 |
|---|---:|---|
| `A_data.bin` | `float32` | 所有非零向量块的数值。每个向量块连续保存 16 个元素，host 读入后转成 fp16。 |
| `A_cols.bin` | `int32` | 每个非零向量块所在的 A 列号，也就是需要读取的 B 行号。 |
| `A_indptr.bin` | `int32` | row block 指针，长度为 `M/16 + 1`。 |

逻辑关系：

```text
第 r 个 row block 的非零向量范围：
A_indptr[r] 到 A_indptr[r+1] - 1

该 row block 的非零向量数：
A_indptr[r+1] - A_indptr[r]
```

`max_vectors_per_row` 是所有 row block 中最大的非零向量数。host 运行时会根据它选择合适的 tiling kernel。

### 3.2 B：稠密 row-major 格式

输入文件 `B_dense.bin` 是 `K × N_original` 的 row-major float32：

```text
B[k, n] = B_dense[k × N_original + n]
```

host 侧会把 B 转成 fp16，并按当前选择的计算粒度对 N 方向做 padding：

```text
N_kernel = 向上取整(N_original, 当前 AIC 一次处理的列数)
```

padding 出来的额外列全部填 0。这样不会改变真实输出，因为额外列对应的 B 值为 0，且验证时只比较原始 N 范围内的列。

### 3.3 C：输出布局

参考结果 `C_dense.bin` 是 row-major float32：

```text
C_ref[i, j] 存在 C_ref[i × N_original + j]
```

kernel 输出 `C_out` 当前为 column-major float32：

```text
C_out[i, j] 存在 C_out[j × M + i]
```

验证时只比较真实列范围：

```text
for i in 0..M-1:
  for j in 0..N_original-1:
    got = C_out[j × M + i]
    ref = C_ref[i × N_original + j]
```

---

## 4. 内核函数接口概述

host 侧通过 `ascblasSpmm(...)` 启动 kernel。接口里包含：

```text
矩阵规模：M、N_kernel、K
A 数据：values、col_indices、row_offsets
B 数据：B_padded
C 输出：C
B pack workspace：用于 AIV 和 AIC 之间传递整理后的 B 数据
调试输出：verify 模式下用于定位错误
```

当前一次编译会生成 4 个特化 kernel。运行时根据 A 每个 row block 的最大非零向量数选择版本：

| 适用范围 | AIC 一次处理的 N 列数 | 稀疏 K 分块大小 | 适用直觉 |
|---:|---:|---:|---|
| 非零向量较少 | 128 | 256 | N 方向展开更大，适合较高稀疏度 |
| 非零向量中等 | 64 | 512 | 平衡 N 展开和 K 容量 |
| 非零向量较多 | 32 | 1024 | 稀疏 K 更长，需要减小 N 批次 |
| 非零向量很多 | 16 | 2048 | 低稀疏度或极端稠密 row block |

这样做的目的不是把所有情况塞进一个完全动态 kernel，而是在保留编译期常量优化的同时，让 host 在运行时选择最合适的特化版本。

---

## 5. 算子整体流水

启动 kernel 前的 host 流程可以概括为三步：

```text
初始化和准备数据
    ↓
对 N 方向做尾块 padding
    ↓
传入 device 数据并启动 kernel
```

完整一点的整体流程如下：

```text
[Host]
  初始化 ACL / device / stream
      │
      ├─ 读取 A 的 vector-CSR 元数据和数值
      ├─ 读取 B，并按 N 方向补零到计算批次对齐
      ├─ 分配 C、workspace、debug buffer
      └─ 把输入数据拷贝到 device
              │
              ▼
[Kernel]
  AIV 侧：根据 A 的列索引 gather B 的若干行，重排为 B_selected^T 所需的连续 workspace
              │
              │ ready/ack 同步
              ▼
  AIC 侧：搬入 B_selected^T 和 A_block^T，执行 MAD，Fixpipe 写回 C
              │
              ▼
[Host]
  拷回 C，verify 或解析 profiler 结果
```

一句话概括：

```text
AIV 负责把“稀疏索引导致的乱序 B 行”整理成“Cube 能连续搬运的左矩阵”；
AIC 负责把整理后的 B 和 A 的向量块做矩阵乘加，并写回输出。
```

---

## 6. AIV 侧流水

AIV 的核心任务是 **B gather + B pack**。

为什么要在 AIV 做这件事？因为 A 的每个非零向量块都有一个列号，这个列号决定要读取 B 的哪一行。如果直接让 AIC 按这些列号去原始 B 中读取，就会出现大量不连续访存，Cube 主计算会被随机访问拖慢。AIV 先把这些 B 行收集并整理好，AIC 后面就可以像处理连续矩阵一样搬运数据。

AIV 侧文本流程图：

```text
当前 row block 的 A 元数据
    │
    ├─ 读取该 row block 的 B 行号列表
    │       这些行号来自 A 的 vector-CSR 列索引
    │
    ├─ 按 N 方向分成若干大列批
    │       每批最多覆盖一段连续 N 列
    │
    ├─ 对每个列批：
    │       从 B 的原始 GM 中 gather 对应行
    │       在 UB 中整理成 row-major packed panel
    │       写入 GM workspace
    │
    ├─ 通知 AIC：当前列批的 B pack 已经准备好
    │
    └─ 等待 AIC 消费完成后，复用 workspace slot
```

AIV 侧输出的 workspace 可以理解为：

```text
workspace 中第 i 行 = 第 i 个非零向量块所需的 B 行片段
workspace 中第 j 列 = 当前 N 列批中的第 j 个输出列
```

也就是：

```text
B_pack[i, j] = B[第 i 个非零向量块对应的 B 行, 当前列批中的第 j 列]
```

这个布局有两个作用：

1. 把原始 B 的随机行访问变成一片连续 workspace；
2. 让 AIC 侧可以直接使用“GM 到 L1，同时做矩阵格式转换”的搬运接口，将 workspace 连续搬入并转成 Cube 友好的格式。

当前正式流水图中不再展示 direct fallback。AIV 侧的主路径就是按组 gather 和 pack B 数据，服务于后续 AIC 的连续搬运与格式转换。

---

## 7. AIC 侧流水

AIC 的核心任务是 **搬入 A 和 B、执行 Cube 乘加、写回 C**。它不是只处理 B；A 和 B 都需要进入 AIC 的 L1/L0 层级，只是二者来源和复用方式不同。

AIC 侧文本流程图：

```text
A 的 vector-CSR 数据
    │
    ├─ 读取当前 row block 的 A values
    ├─ 构造 A_block
    ├─ 搬运 A values 到 L1_B
    └─ 搬运/转置到 L0B，形成 A_block^T

B pack workspace
    │
    ├─ 等待 AIV ready
    ├─ 从 workspace 连续搬运 B pack 到 L1_A
    ├─ 搬运过程中做 ND → NZ 类格式转换
    └─ 搬运/转置到 L0A，形成 B_selected^T

L0A = B_selected^T
L0B = A_block^T
    │
    ├─ MAD 矩阵乘加
    └─ Fixpipe 将 L0C 写回 GM C
```

更贴近执行顺序的简略流程：

```text
for 每个 row block:
  读取该 row block 的 A 非零向量块
  将 A_block^T 搬入并尽量驻留在 L0B

  for 每个 N 列批:
    等待 AIV 完成 B pack
    将 B pack 连续搬入 L1_A，并转为 Cube 友好格式
    将 B_selected^T 搬入 L0A
    执行 MAD: B_selected^T × A_block^T
    Fixpipe 写回 C 的对应 16 行 × 当前列批
    通知 AIV 该 B pack 可以复用
```

### 7.1 A 矩阵在 AIC 中的角色

对于一个 row block，A 的 16 行与若干非零列构成：

```text
A_block: 16 × 非零向量数
```

Cube 实际需要右矩阵：

```text
A_block^T: 非零向量数 × 16
```

AIC 会把 A 的 values 从 GM 搬入 L1，再搬到 L0B。对于非零向量数不超过当前稀疏 K 分块大小的常见场景，A_block^T 可以在一个 row block 的多个 N 列批之间复用，减少重复搬运。

### 7.2 B 矩阵在 AIC 中的角色

B 已经由 AIV 按当前 row block 的列索引打包到 workspace。AIC 不再从原始 B 中随机 gather，而是从 workspace 连续搬运。这个设计的关键收益是：

```text
AIV 先整理 B 行
    ↓
AIC 使用连续矩阵搬运 + 格式转换接口
    ↓
L1/L0 中得到 B_selected^T
    ↓
Cube 直接执行矩阵乘加
```

这里使用的是 ND 到 NZ/fractal 一类的矩阵搬运与格式转换路径。实现中可能通过封装函数调用，底层对应 `copy_gm_to_cbuf_multi_nd2nz_b16` 这类接口。它的价值在于：**一次完成连续拷贝和 Cube 友好布局转换**，避免 AIC 内部再做逐行 gather 或手动转置。

---

## 8. 同步操作

该 kernel 同时使用两类同步：

### 8.1 AIV 与 AIC 之间的生产者-消费者同步

AIV 生产 B pack，AIC 消费 B pack。二者通过 workspace 和 ready/ack 标志协同：

```text
AIV 写入 workspace slot
    ↓
AIV 发送 ready
    ↓
AIC 等待 ready
    ↓
AIC 从该 slot 读取 B pack 并完成后续计算
    ↓
AIC 发送 ack
    ↓
AIV 收到 ack 后复用这个 slot
```

这保证了 AIV 不会覆盖 AIC 尚未消费完的 workspace 数据。

### 8.2 AIC 内部流水同步

AIC 内部还需要保证搬运、加载、计算、写回之间的顺序：

```text
GM → L1 搬运完成
    ↓
L1 → L0 搬运完成
    ↓
MAD 计算完成
    ↓
Fixpipe 写回完成
```

这些同步通常由 CCE Intrinsic 的 pipe flag 机制完成。对于初学者，可以把它理解成：每一级硬件流水都需要确认上一阶段数据已经准备好，才能安全启动下一阶段。

---

## 9. Host 侧操作概述

Host 侧负责准备输入、选择 kernel、启动计算和处理结果。较完整流程如下：

```text
1. 初始化 ACL、device、stream、handle
2. 读取 A 的 vector-CSR 数据
3. 统计每个 row block 的非零向量数，得到最大值
4. 根据最大非零向量数选择 4 个特化 kernel 中的一个
5. 根据所选 kernel 的 N 批次粒度，对 N 方向做 padding
6. 读取 B，并构造补零后的 B_padded
7. 计算每个 row block 在 B-pack workspace 中的偏移
8. 分配 device 内存和 workspace
9. 拷贝 A、B、元数据到 device
10. 启动 kernel
11. verify 模式：拷回 C，与参考 C 比较
12. prof 模式：连续运行并由 profiler 统计平均耗时
13. 清理资源
```

Host 侧选择 kernel 的逻辑可以概括为：

```text
如果某个 row block 最多只有较少非零向量：
    选择 N 方向更宽的 kernel
否则：
    减小 N 方向批次，增大稀疏 K 方向容量
```

Host 侧 padding 逻辑：

```text
N_kernel = 向上取整(N_original, 当前 kernel 一次处理的 N 列数)
B_padded[:, 0:N_original] = 原始 B
B_padded[:, N_original:N_kernel] = 0
```

这样做让 AIC 避免处理非满 N 批次，同时不改变真实输出。

---

## 10. 使用到的优化方法

### 10.1 C 核 / V 核协同

AIV 更适合处理 gather、pack、数据重排等向量和搬运任务；AIC 更适合执行矩阵乘加。当前算子将不规则 B 行访问放到 AIV，把规整矩阵计算交给 AIC，降低 Cube 主路径的随机访存压力。

### 10.2 vector-CSR 稀疏格式

A 使用 16 行竖直向量块作为非零单元。这种格式天然匹配输出的 16 行 C block，也便于把 SpMM 转换为小型 GEMM：

```text
16 行 A 块 × 若干 B 行 = 16 行 C 块
```

### 10.3 多 tiling 特化与运行时 dispatch

不同稀疏度下，每个 row block 的非零向量数差异很大。当前一次编译 4 个特化 kernel，运行时根据 A 的统计信息选择。这样兼顾了通用性和性能。

### 10.4 B 数据打包与连续格式转换

AIV 把 `B[col_indices, n_range]` 提前整理到连续 workspace。AIC 随后可以使用矩阵搬运 + 格式转换接口，将这块连续数据直接搬到 L1 并变成 Cube 友好的布局。这是当前版本的重要优化点。

### 10.5 向量化数据传输

AIV 侧以成组方式处理多个非零向量块，使用 UB 作为中转，减少小粒度搬运；AIC 侧使用矩阵搬运接口，将连续 workspace 高效搬入 L1/L0。

### 10.6 A 块复用/常驻

对于一个 row block，A_block^T 在多个 N 列批之间不变。因此 AIC 会尽量将 A_block^T 保留在更靠近计算单元的缓存层级中，让多个 N 批次复用，减少 A 侧重复搬运。

### 10.7 不规则 N padding

N 不要求为 16 或 128 的倍数。Host 会根据当前选择的 kernel，把 N padding 到该 kernel 的 N 批次粒度。padding 列填 0，数学结果不变，同时能显著降低 AIC/Fixpipe 处理非满列批时的损失。

### 10.8 K 尾块处理

K 不要求为 16 的倍数。由于实际计算沿 A 的非零向量块进行，kernel 内部区分真实非零向量数和硬件对齐后的长度，保证 K 方向尾块可以正确处理。

### 10.9 Workspace 环形缓冲

AIV 与 AIC 通过 workspace 传递 B pack。workspace 被划分成若干 slot，AIV 和 AIC 通过 ready/ack 控制复用，使生产和消费能够重叠。

---

## 11. 其他操作脚本作用

### 11.1 `spmm_gen_data.py`

生成单个临时测试数据集，包括：

```text
A_dense.bin
B_dense.bin
C_dense.bin
csr/A_*.bin
vector_csr/A_*.bin
```

当前算子实际使用的是 `vector_csr/` 下的数据；标准 CSR 和 A_dense 主要用于备用、调试或生成参考结果。

### 11.2 `generate_full_datasets.py`

根据 JSON case 列表批量生成可复用完整数据集，默认输出到：

```text
ascblas/datasets_full/<case_name>/
```

每个 case 包含完整 A/B/C、CSR、vector-CSR 和 `manifest.json`。

### 11.3 `use_full_dataset.py`

将某个 full dataset 拷贝到当前运行目录：

```text
ascblas/datasets_full/<case_name> -> ascblas/data
```

`run.sh` 和 `test.sh` 都会使用它切换数据。

### 11.4 `run.sh`

单 case 运行脚本，支持：

```text
gen / verify / prof
```

参数格式：

```bash
./run.sh <M> <N> <K> [sparsity] [d] [mode] [device_id] [rebuild]
```

其中：

```text
rebuild=1: 重新编译，默认
rebuild=0: 复用已有 build
```

### 11.5 `test.sh`

多 case profiling 脚本，读取 JSON case 列表，一次编译或复用已有编译结果，逐个 case 调用 msprof，并汇总到：

```text
ascblas/shell/prof_multi_cases/summary.tsv
```

参数格式：

```bash
./test.sh <cases_json> [device_id] [rebuild] [case_name ...]
```

### 11.6 `prof.py`

解析 msprof 生成的 `task_time_*.csv`，默认跳过第一次 warmup，统计后 20 次 kernel 平均时间，并输出：

```text
M N K sparsity avg_ms effective_tflops dense_equiv_tflops
```

其中：

```text
dense_equiv_tflops = 2 × M × N × K / time
effective_tflops   = 2 × nnz(A) × N / time
```

如果能读取 `A_data.bin`，则使用实际非零元素数；否则按稀疏度估算。

---

## 12. 运行方法

### 12.1 单 case verify

```bash
cd ascblas/shell

SPMM_FULL_DATA_CASE=bench_sq_m4096_n4096_k4096_s95_d16 \
./run.sh 4096 4096 4096 95 16 verify 0 1
```

复用已有 build：

```bash
SPMM_FULL_DATA_CASE=bench_sq_m4096_n4096_k4096_s95_d16 \
./run.sh 4096 4096 4096 95 16 verify 0 0
```

### 12.2 单 case profile

```bash
SPMM_FULL_DATA_CASE=bench_sq_m4096_n4096_k4096_s95_d16 \
./run.sh 4096 4096 4096 95 16 prof 0 0
```

### 12.3 批量生成 full dataset

```bash
cd ascblas/tools

python3 generate_full_datasets.py \
  --cases spmm_report_benchmark_cases_v2.json
```

只生成某几个 case：

```bash
python3 generate_full_datasets.py \
  --cases spmm_report_benchmark_cases_v2.json \
  --only bench_sq_m4096_n4096_k4096_s95_d16 bench_stress_m8192_n8192_k8192_s95_d16
```

### 12.4 批量 profile

第一次运行，重新编译一次：

```bash
cd ascblas/shell
./test.sh ../tools/spmm_report_benchmark_cases_v2.json 0 1
```

后续复用 build：

```bash
./test.sh ../tools/spmm_report_benchmark_cases_v2.json 0 0
```

只跑部分 case：

```bash
./test.sh ../tools/spmm_report_benchmark_cases_v2.json 0 0 \
  bench_sq_m4096_n4096_k4096_s95_d16 \
  bench_stress_m8192_n8192_k8192_s95_d16
```

结果文件：

```text
ascblas/shell/prof_multi_cases/summary.tsv
```

---

## 13. 910B 测试结果总结

测试设备：Ascend 910B。  
测试数据：`bench_` 系列 38 个 case，包含 6 个方阵规模、5 档稀疏度、1 个 8192 压力测试、6 个长方体 case、1 个特殊不规则 case。

### 13.1 代表性结果

| case | shape | sparsity | avg_ms | effective TFLOPS | dense-equivalent TFLOPS | tiling |
|---|---:|---:|---:|---:|---:|---|
| `bench_sq_m4096_n4096_k4096_s95_d16` | 4096³ | 95 | 0.754697 | 9.105574 | 182.111435 | AIC8 / K256 |
| `bench_stress_m8192_n8192_k8192_s95_d16` | 8192³ | 95 | 6.391463 | 8.601408 | 172.028161 | AIC4 / K512 |
| `bench_sq_m4096_n4096_k4096_s90_d16` | 4096³ | 90 | 1.681949 | 8.171412 | 81.714103 | AIC4 / K512 |
| `bench_sq_m3072_n3072_k3072_s95_d16` | 3072³ | 95 | 0.384485 | 7.540221 | 150.804475 | AIC8 / K256 |
| `bench_special_m4448_n3333_k5555_s95_d16` | 4448×3333×5555 | 95 | 1.201090 | 6.856597 | 137.131934 | AIC4 / K512 |

### 13.2 方阵规模趋势

以 95% 稀疏度为例：

| size | avg_ms | effective TFLOPS | dense-equivalent TFLOPS |
|---:|---:|---:|---:|
| 256 | 0.010706 | 0.156718 | 3.134171 |
| 512 | 0.016274 | 0.824725 | 16.494744 |
| 1024 | 0.047478 | 2.261565 | 45.231131 |
| 2048 | 0.152248 | 5.642062 | 112.841346 |
| 3072 | 0.384485 | 7.540221 | 150.804475 |
| 4096 | 0.754697 | 9.105574 | 182.111435 |

小规模下固定开销占比较高；规模增大后，pipeline 与 Cube 计算利用率提高，性能逐渐接近高效区间。

### 13.3 稀疏度趋势

以 4096³ 为例：

| sparsity | avg_ms | effective TFLOPS | dense-equivalent TFLOPS | tiling |
|---:|---:|---:|---:|---|
| 85 | 3.575359 | 5.766089 | 38.440602 | AIC2 / K1024 |
| 90 | 1.681949 | 8.171412 | 81.714103 | AIC4 / K512 |
| 95 | 0.754697 | 9.105574 | 182.111435 | AIC8 / K256 |
| 98 | 0.475549 | 5.780217 | 289.011129 | AIC8 / K256 |
| 99 | 0.372081 | 3.693788 | 369.379123 | AIC8 / K256 |

结论：dense-equivalent TFLOPS 会随稀疏度提升而增加；effective TFLOPS 在 90%~95% 附近较优，极高稀疏度下由于实际计算量变少，固定开销占比增大，effective TFLOPS 反而下降。

### 13.4 长方体与特殊规模

| case | shape | sparsity | avg_ms | effective TFLOPS | dense-equivalent TFLOPS |
|---|---:|---:|---:|---:|---:|
| `bench_rect_tfdown_m1024_n2048_k4096_s95_d16` | 1024×2048×4096 | 95 | 0.130236 | 6.595662 | 131.913366 |
| `bench_rect_bert_down_m768_n1024_k3072_s95_d16` | 768×1024×3072 | 95 | 0.055801 | 4.329534 | 86.590531 |
| `bench_rect_tfup_m4096_n2048_k1024_s95_d16` | 4096×2048×1024 | 95 | 0.226741 | 3.788431 | 75.768693 |
| `bench_rect_rnn_m2048_n128_k8192_s95_d16` | 2048×128×8192 | 95 | 0.120701 | 1.779177 | 35.583527 |
| `bench_special_m4448_n3333_k5555_s95_d16` | 4448×3333×5555 | 95 | 1.201090 | 6.856597 | 137.131934 |

N 较小的 case，例如 `N=128` 的 RNN 形状，AIC N batch 难以充分展开，固定开销不易摊薄，性能明显低于大 N 场景。特殊不规则 case 通过测试，说明 N/K 非规则输入支持已经具备实用性。

---

5. **性能报告自动化**：可将 `summary.tsv` 自动转为表格和图表，用于汇报材料生成。
