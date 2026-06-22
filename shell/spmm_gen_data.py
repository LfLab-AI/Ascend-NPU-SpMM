#!/usr/bin/env python3
################################################################################
# SPMM (Sparse Matrix Multiplication) Data Generator for SR-BCRS format
#
# Generates test data for SR-BCRS-format SPMM kernel
# Creates: vector_csr format for A, dense format for B and C
#
# Usage: python3 spmm_gen_data.py <M> <N> <K> [sparsity] [d]
#   M, N, K: Matrix dimensions (A: M×K, B: K×N, C: M×N)
#   sparsity: Sparsity percentage for A (default: 85)
#   d: Vector block dimension (default: 16, should match vec_length in kernel)
################################################################################

import sys
import os
import numpy as np
from scipy.sparse import csr_matrix

def write_bin_file(file_path, data):
    """Write numpy array to binary file"""
    data.tofile(file_path)
    print(f"✅ 生成二进制文件: {file_path} ({data.shape}, {data.dtype})")

def generate_1d_vector_sparse_matrix(rows, cols, sparsity_percent, d):
    """
    生成A矩阵：1-D结构化稀疏矩阵（nx1向量块，d维垂直向量）
    - 向量块: 同一列，连续d行 (1×d)
    - 起始行必须是 d 的倍数 (0, d, 2d, ...)
    - 保证生成的向量块唯一，无重复位置
    """
    mat = np.zeros((rows, cols), dtype=np.float32)
    total_elements = rows * cols
    target_non_zero = total_elements * (1 - sparsity_percent / 100)

    # 合法起始行：必须是 d 的倍数，且保证不越界
    valid_start_rows = [r for r in range(0, rows - d + 1, d)]
    if not valid_start_rows:
        raise ValueError(f"行数{rows}不足，无法生成d={d}的向量块（起始行需为{d}的倍数）")

    max_possible_blocks = len(valid_start_rows) * cols
    num_blocks = max(min(int(target_non_zero // d), max_possible_blocks), 1)

    vector_info = []
    generated_positions = set()  # 确保向量块位置唯一

    for _ in range(num_blocks):
        # 生成唯一的向量块位置
        while True:
            start_row = np.random.choice(valid_start_rows)
            start_col = np.random.randint(0, cols)
            pos_key = (start_row, start_col)
            if pos_key not in generated_positions:
                generated_positions.add(pos_key)
                break

        end_row = start_row + d - 1
        # 生成d维向量并填充到同一列的连续d行
        vector_vals = np.random.randn(d).astype(np.float32)
        for i in range(d):
            mat[start_row + i, start_col] = vector_vals[i]

        vector_info.append({
            "start_row": start_row,
            "end_row": end_row,
            "col": start_col,
            "values": vector_vals
        })

    # 按起始行、列排序（满足验证时的顺序要求）
    vector_info.sort(key=lambda x: (x["start_row"], x["col"]))
    return mat, vector_info

def generate_dense_matrix(rows, cols):
    """
    生成B矩阵：纯密集矩阵（无稀疏，随机值）
    按普通矩阵行优先格式存储
    """
    mat = np.random.randn(rows, cols).astype(np.float32)
    return mat

def save_csr(mat, csr_mat, save_dir, mat_name):
    """保存A矩阵的标准CSR格式（备用）"""
    os.makedirs(save_dir, exist_ok=True)
    csr_bin_dir = os.path.join(save_dir, "csr")
    os.makedirs(csr_bin_dir, exist_ok=True)

    # 保存标准CSR三数组
    csr_mat.data.astype(np.float32).tofile(os.path.join(csr_bin_dir, f"{mat_name}_data.bin"))
    csr_mat.indices.astype(np.int32).tofile(os.path.join(csr_bin_dir, f"{mat_name}_indices.bin"))
    csr_mat.indptr.astype(np.int32).tofile(os.path.join(csr_bin_dir, f"{mat_name}_indptr.bin"))

    # 保存标准CSR元信息
    meta_path = os.path.join(csr_bin_dir, f"{mat_name}_csr_meta.txt")
    with open(meta_path, 'w') as f:
        f.write(f"shape: {mat.shape[0]} {mat.shape[1]}\n")
        f.write(f"non_zero_count: {csr_mat.nnz}\n")
    print(f"✅ 标准CSR元信息已保存: {meta_path}")

def save_vector_csr(vector_info, d, rows, save_dir, mat_name):
    """
    保存A矩阵的自定义vector-CSR格式：
    - data: 同一d维向量块数据相邻存储
    - 列索引数组: n个元素（n为向量块数），每个元素是向量块所在列
    - 行指针数组: 长度=M/d+1，indptr[i+1]-indptr[i]为第i*d~(i+1)*d行的非零向量块数
    """
    os.makedirs(save_dir, exist_ok=True)
    vector_csr_dir = os.path.join(save_dir, "vector_csr")
    os.makedirs(vector_csr_dir, exist_ok=True)

    # 1. 提取vector-CSR核心数据
    vector_data = []       # 向量块数值（d个相邻）
    vector_cols = []       # 向量块列索引（n个）
    vector_start_rows = [] # 向量块起始行

    for vec in vector_info:
        vector_data.extend(vec["values"])
        vector_cols.append(vec["col"])
        vector_start_rows.append(vec["start_row"])

    # 2. 生成vector-CSR行指针（核心：长度=M/d+1）
    num_row_blocks = rows // d  # M/d个d行块
    vec_csr_indptr = [0] * (num_row_blocks + 1)
    current_count = 0

    for i in range(num_row_blocks):
        # 计算第i个d行块（i*d ~ (i+1)*d -1）的非零向量块数
        start_row_block = i * d
        end_row_block = (i + 1) * d - 1
        block_vec_count = 0

        for vec in vector_info:
            if vec["start_row"] >= start_row_block and vec["start_row"] <= end_row_block:
                block_vec_count += 1

        current_count += block_vec_count
        vec_csr_indptr[i+1] = current_count

    # 3. 保存vector-CSR数据
    vector_data_np = np.array(vector_data, dtype=np.float32)
    vector_cols_np = np.array(vector_cols, dtype=np.int32)
    vec_csr_indptr_np = np.array(vec_csr_indptr, dtype=np.int32)

    vector_data_np.tofile(os.path.join(vector_csr_dir, f"{mat_name}_data.bin"))
    vector_cols_np.tofile(os.path.join(vector_csr_dir, f"{mat_name}_cols.bin"))
    vec_csr_indptr_np.tofile(os.path.join(vector_csr_dir, f"{mat_name}_indptr.bin"))

    # 4. 保存vector-CSR元信息
    meta_path = os.path.join(vector_csr_dir, f"{mat_name}_vector_csr_meta.txt")
    with open(meta_path, 'w') as f:
        f.write(f"shape: {rows} {vector_info[0]['end_row']+1 if vector_info else 0}\n")
        f.write(f"d: {d}\n")
        f.write(f"num_vector_blocks: {len(vector_info)}\n")
        f.write(f"row_blocks_count: {num_row_blocks}\n")
    print(f"✅ vector-CSR元信息已保存: {meta_path}")

def save_dense_matrix_bin(mat, save_dir, mat_name):
    """保存密集矩阵（B矩阵和C矩阵专用）"""
    os.makedirs(save_dir, exist_ok=True)
    dense_bin_path = os.path.join(save_dir, f"{mat_name}_dense.bin")
    mat.astype(np.float32).flatten(order='C').tofile(dense_bin_path)

    # 保存密集矩阵元信息
    meta_path = os.path.join(save_dir, f"{mat_name}_dense_meta.txt")
    with open(meta_path, 'w') as f:
        f.write(f"shape: {mat.shape[0]} {mat.shape[1]}\n")
    print(f"✅ 密集矩阵元信息已保存: {meta_path}")

def print_vector_blocks_sorted(vector_info, mat_name):
    """
    按行数从小到大、列数从小到大打印向量块（满足验证顺序要求）
    """
    print(f"\n=== {mat_name}矩阵 - 向量块信息（按行、列排序）===")
    if not vector_info:
        print("⚠️  无向量块数据")
        return

    # 二次排序确保顺序正确（起始行升序 → 列升序）
    sorted_vector_info = sorted(vector_info, key=lambda x: (x["start_row"], x["col"]))

    for idx, vec in enumerate(sorted_vector_info):
        vals_str = ", ".join([f"{v:.4f}" for v in vec["values"]])
        print(f"向量块 #{idx+1}: 行[{vec['start_row']}~{vec['end_row']}], 列={vec['col']}, 值=[{vals_str}]")
        if idx > 3 :
            return
        

def generate_spmm_data(M, N, K, sparsity, d, save_dir):
    """
    生成SPMM测试数据（SR-BCRS格式）
    A: M×K 稀疏矩阵（1-D向量块结构化稀疏）
    B: K×N 密集矩阵
    C: M×N 结果矩阵
    """
    print("\n" + "="*60)
    print("SPMM SR-BCRS格式数据生成器")
    print("="*60)
    print(f"矩阵维度: M={M}, N={N}, K={K}")
    print(f"稀疏度: {sparsity}% (A矩阵)")
    print(f"向量块维度: d={d}")
    print(f"输出目录: {save_dir}")

    # ====================== 处理矩阵A (1-D结构化稀疏) ======================
    print("\n" + "="*60)
    print(f"=== 处理矩阵A ({M}×{K}) - 1-D结构化稀疏 ===")
    print("="*60)
    
    # 1. 生成A矩阵
    print("\n【1/5】生成1-D结构化稀疏矩阵A")
    A, A_vector_info = generate_1d_vector_sparse_matrix(M, K, sparsity, d)
    print("原始矩阵A:")
    print(np.round(A, 4))

    # 2. 保存标准CSR格式（备用）
    print("\n【2/5】保存标准CSR格式")
    A_csr = csr_matrix(A)
    save_csr(A, A_csr, save_dir, "A")

    # 3. 保存自定义vector-CSR格式（SR-BCRS使用）
    print("\n【3/5】保存自定义vector-CSR格式")
    save_vector_csr(A_vector_info, d, M, save_dir, "A")

    # 4. 按行、列排序打印向量块（验证顺序）
    print("\n【4/5】验证向量块顺序（行升序→列升序）")
    print_vector_blocks_sorted(A_vector_info, "A")

    # 5. 保存A矩阵密集格式（备用）
    print("\n【5/5】保存A矩阵密集格式")
    save_dense_matrix_bin(A, save_dir, "A")

    # ====================== 处理矩阵B (纯密集矩阵) ======================
    print("\n" + "="*60)
    print(f"=== 处理矩阵B ({K}×{N}) - 纯密集矩阵 ===")
    print("="*60)
    # 1. 生成B矩阵
    print("\n【1/2】生成纯密集矩阵B")
    B = generate_dense_matrix(K, N)
    print("原始矩阵B:")
    print(np.round(B, 4))

    # 2. 保存B矩阵密集格式
    print("\n【2/2】保存B矩阵密集格式")
    save_dense_matrix_bin(B, save_dir, "B")

    # ====================== 生成矩阵乘积C=A*B（参考结果） ======================
    print("\n" + "="*60)
    print("=== 生成参考结果 C = A * B ===")
    print("="*60)
    C = np.dot(A, B).astype(np.float32)
    save_dense_matrix_bin(C, save_dir, "C")
    print("参考结果矩阵C:")
    print(np.round(C, 4))

    # ====================== 文件目录说明 ======================
    print("\n" + "="*60)
    print("=== 生成的文件目录结构 ===")
    print("="*60)
    print(f"{save_dir}/")
    print("├── csr/      # A矩阵标准CSR格式")
    print("│   ├── A_data.bin     # CSR数值")
    print("│   ├── A_indices.bin  # CSR列索引")
    print("│   ├── A_indptr.bin   # CSR行指针")
    print("│   └── A_csr_meta.txt # A矩阵CSR元信息")
    print("├── vector_csr/        # A矩阵自定义vector-CSR格式（SR-BCRS使用）")
    print("│   ├── A_data.bin     # 向量块数值（d个相邻）")
    print("│   ├── A_cols.bin     # 向量块列索引（n个）")
    print("│   ├── A_indptr.bin   # 行指针（M/d+1个）")
    print("│   └── A_vector_csr_meta.txt    # A矩阵vector-CSR元信息")
    print("├── A_dense.bin              # A矩阵密集格式")
    print("├── A_dense_meta.txt         # A矩阵密集格式元信息")
    print("├── B_dense.bin              # B矩阵密集格式")
    print("├── B_dense_meta.txt         # B矩阵密集格式元信息")
    print("├── C_dense.bin              # 参考结果C=A*B")
    print("└── C_dense_meta.txt         # 参考结果元信息")
    print("\n✅ 数据生成完成！")

if __name__ == "__main__":
    # 解析命令行参数
    if len(sys.argv) < 4:
        print("Usage: python3 spmm_gen_data.py <M> <N> <K> [sparsity] [d]")
        print("  M, N, K: Matrix dimensions")
        print("  sparsity: Sparsity percentage for A (default: 85)")
        print("  d: Vector block dimension (default: 16)")
        print("\nExample:")
        print("  python3 spmm_gen_data.py 256 256 256")
        print("  python3 spmm_gen_data.py 256 256 256 90")
        print("  python3 spmm_gen_data.py 256 256 256 90 16")
        sys.exit(1)

    M = int(sys.argv[1])
    N = int(sys.argv[2])
    K = int(sys.argv[3])
    sparsity = int(sys.argv[4]) if len(sys.argv) > 4 else 85
    d = int(sys.argv[5]) if len(sys.argv) > 5 else 16

    # 参数检查
    if M % d != 0:
        print(f"Error: M ({M}) must be divisible by d ({d})")
        sys.exit(1)
    if sparsity < 0 or sparsity >= 100:
        print(f"Error: Sparsity {sparsity} must be between 0 and 99")
        sys.exit(1)

    save_dir = "../data"  # 相对于shell目录的数据保存路径
    generate_spmm_data(M, N, K, sparsity, d, save_dir)