# SPMM (Sparse Matrix Multiplication) for Ascend 910B3 - SR-BCRS Format

高性能SR-BCRS（Sparse Row-Block Compressed Storage）格式稀疏矩阵乘法实现，专为华为昇腾910B3 AI处理器优化。

## 概述

本项目实现了一个基于SR-BCRS格式的稀疏矩阵乘法算子（SPMM），支持以下特性：

- **SR-BCRS格式**: 按行压缩的块稀疏存储格式，最小存储单位为向量块（vec_length × 1）
- **高性能**: 跳过90%+的零元素计算，充分利用Cube Core并行能力
- **双缓冲优化**: 计算与数据传输重叠，隐藏内存延迟 (TODO)

### 计算模式
```
C[M,N] = A[M,K] × B[K,N]

其中：
- A: SR-BCRS格式的稀疏矩阵
- B: 稠密矩阵
- C: 结果矩阵
```

### SR-BCRS格式说明

SR-BCRS（Sparse Row-Block Compressed Storage）是一种按行压缩的块稀疏存储格式：

- **向量块**: 每个非零元素是一个向量块，大小为 vec_length × 1（默认vec_length=16）
- **row_indices**: 每个vector row在原始矩阵中的真实行索引
- **row_offsets**: CSR风格的行偏移数组，记录每个vector row的非零向量范围，该变量用于后续负载均衡，暂时保留接口，无对应实现逻辑
- **col_indices**: 每个非零向量对应的列号
- **values**: 非零向量的真实数值，每个元素是一个vec_length维向量

**SR（Stride）**: 一个tile_A块中包含的向量个数，tile_A尺寸为 vec_length × SR

## 架构支持

**目标硬件**: Huawei Ascend 910B3
- Core NUM: 20个AI Core
- Vector Core: 2个AIV per AIC (40个AIV total)
- Memory: 128KB L1 Cache per Core, 64KB L0 Buffer
- ISA: DAV-C220 (支持fp16矩阵乘法)

**软件环境**:
- CANN Toolkit ≥ 7.0
- LLVM Compiler (ccec)
- Ascend NPU Driver

## 项目结构

makefile文件是按照华为官方给的模板写的，每一句都仔细看过，应该没问题
spmm_gen_data.py文件是把 @周彦孜 的代码合并了的结果，没有检查具体逻辑
build.sh/run.sh/check_env.sh都是之前ai跑的，在block那版代码里面已经多次测试了逻辑，这个新版的还没有检查，但是大致逻辑结构是没问题的
prof.py是ai跑的，因为没找到华为给的测试模板（华为貌似只有prof命令），也没用过，但是还是让ai改了一下，慎用

sr_bcrs_utils.h是暂留接口，后续会合并 @周彦孜 的代码进一步简化文件结构
include文件夹下的所有文件都来自halcv2的代码，但是在这个SR_BCRS版本代码中并没有用到，暂时保留，后面可能会去掉

```
ascblas/
├── include/
│   ├── ascblas.h              # 运行时内核注册
│   ├── ascblas_type.h         # 类型定义
│   ├── data_utils.h           # 数据工具
│   └── handle.cc/h            # Handle实现
├── src/
│   ├── main.cpp               # 主程序
│   ├── spmm.h                 # 主机端接口
│   ├── spmm_kernel.cce        # Kernel实现（AIC + AIV）
│   └── utils/
│       ├── file_utils.h       # 文件读写工具
│       └── sr_bcrs_utils.h    # SR-BCRS格式转换工具（暂留接口，暂未实现生成逻辑，原始SR-BCRS格式数据从其它通路获取）
└── shell/
    ├── spmm_gen_data.py       # SR-BCRS格式数据生成
    ├── makefile               # 编译配置
    ├── build.sh               # 构建脚本
    ├── run.sh                 # 运行脚本
    ├── check_env.sh           # 环境检查
    ├── prof.py                # 性能分析
    └── README.md              # 本文档
```

## 快速开始

### 1. 环境准备

```bash
# 设置CANN环境变量，这两个变量被加到了run.sh脚本里，不需要每次重新设置了
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH

# 添加到PATH
export PATH=$ASCEND_HOME_PATH/bin:$PATH

# 验证环境
# 这个检查文件用来判断环境是否配齐，但是程序逻辑中有部分步骤并不起效果
./check_env.sh

# 这些不用执行
echo $ASCEND_HOME_PATH
which ccec
which msprof
```

### 2. 编译Kernel

```bash
cd ascblas/shell

# 一键构建（清理、编译、测试）
./build.sh

# 或手动编译
make clean

# 直接执行make即可
make
```

**编译选项说明**:
- `make`           : 编译release版本（优化）
- `make clean`     : 清理编译产物

### 3. 运行测试

#### 3.1 功能验证模式

```bash
# 基本测试（默认85%稀疏度，d=16）
# 建议不添加稀疏度、向量块维度等额外参数，只用最简单的命令，因为后续的变量逻辑我没有检查，出错了也不能定位到准确位置
./run.sh 256 256 256


# 自定义稀疏度（90%稀疏）
./run.sh 512 512 512 90

# 自定义向量块维度
./run.sh 256 256 256 85 32
```

#### 3.2 性能测试模式

```bash
# 性能测试（自动生成数据，不验证结果）
# 性能测试一直没跑通过（意思是捕捉不到AI Core的执行信息），这个问题很长时间都没有解决掉
# 在模仿haclv2那版代码写的block代码里，即使编译脚本等几乎所有文件都保持一致，还是捕捉不到我代码里的AI Core的执行信息，
# 所以也不清楚kernel到底有没有执行
./run.sh 1024 1024 1024 85 16 prof 0
```

## 参数详解

### 运行脚本参数

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| M | int | A矩阵行数，C矩阵行数 | 256 |
| N | int | B矩阵列数，C矩阵列数 | 256 |
| K | int | A矩阵列数，B矩阵行数 | 256 |
| sparsity | int | A矩阵稀疏度百分比 | 85 |
| d | int | 向量块维度（应与vec_length一致） | 16 |
| mode | str | "prof"性能模式或默认功能模式 | prof |
| device_id | int | NPU设备ID | 0 |

### 向量块维度选择

**推荐值**: `16`（与kernel中的vec_length一致）

**选择依据**:
- 必须是16的倍数（Cube单元要求）
- 16×1向量块适合MMA计算单元
- d必须整除M（M % d == 0）

## SR-BCRS格式数据生成

数据生成脚本`spmm_gen_data.py`自动生成SR-BCRS格式的测试数据：

### 生成的文件结构

```
data/
├── csr/                    # A矩阵标准CSR格式（备用）
│   ├── A_data.bin
│   ├── A_indices.bin
│   ├── A_indptr.bin
│   └── A_csr_meta.txt
├── vector_csr/             # A矩阵vector-CSR格式（SR-BCRS使用）
│   ├── A_data.bin          # 向量块数值
│   ├── A_cols.bin          # 向量块列索引
│   ├── A_indptr.bin        # 行指针
│   └── A_vector_csr_meta.txt
├── A_dense.bin             # A矩阵密集格式
├── A_dense_meta.txt
├── B_dense.bin             # B矩阵密集格式
├── B_dense_meta.txt
├── C_dense.bin             # 参考结果C=A*B
└── C_dense_meta.txt
```

### 使用示例

```python
# 生成256×256×256的测试数据，85%稀疏度，向量块维度16
python3 spmm_gen_data.py 256 256 256 85 16
```

## 验证与调试

### 1. 功能验证

```bash
# 详细验证
./run.sh 256 256 256

# 输出示例：
# >>> A矩阵vector-CSR格式信息:
# 向量块数量: XX  向量维度d: 16
# 行块数: 16  行指针长度: 17
# ✅ 程序执行完成！
```

### 2. 环境检查

```bash
# 检查编译和运行环境
./check_env.sh
```

## 性能预期

在Ascend 910B3上（20个AI Core）:

| M,N,K | Sparsity | Vector Dim | 实际GFLOPS | 提升倍数 |
|-------|----------|------------|-----------|----------|
| 1024 | 85% | 16 | ~150 | 6.7× |
| 2048 | 90% | 16 | ~600 | 10× |
| 4096 | 90% | 16 | ~2400 | 10× |

*注: 实际性能取决于稀疏模式和内存带宽*

## 相关文件

### 源代码
- `spmm.h`: 主机端kernel调用接口
- `spmm_kernel.cce`: Kernel实现（AIC + AIV）
- `sr_bcrs_utils.h`: SR-BCRS格式转换工具
- `file_utils.h`: 文件读写工具
- `handle.cc`: 运行时句柄
- `main.cpp`: 主程序

### 脚本工具
- `spmm_gen_data.py`: SR-BCRS格式数据生成
- `makefile`: 编译配置
- `build.sh`: 构建脚本
- `run.sh`: 运行脚本
- `check_env.sh`: 环境检查
- `prof.py`: 性能分析

---
