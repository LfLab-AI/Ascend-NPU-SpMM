#ifndef FILE_UTILS_H
#define FILE_UTILS_H

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>

/**
 * @brief 通用二进制文件读取函数（支持任意类型）
 * @tparam T 数据类型
 * @param file_path 文件路径
 * @param data 输出数据指针
 * @param count 要读取的元素个数
 */
template <typename T>
void ReadBinFile(const std::string &file_path, T *data, size_t count)
{
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open())
    {
        std::cerr << "Error: 无法打开二进制文件 " << file_path << std::endl;
        exit(1);
    }

    file.read(reinterpret_cast<char *>(data), count * sizeof(T));
    if (!file)
    {
        std::cerr << "Error: 读取文件 " << file_path << " 失败，读取字节数: " << file.gcount()
                  << " 期望字节数: " << count * sizeof(T) << std::endl;
        file.close();
        exit(1);
    }

    file.close();
    std::cout << "✅ 成功读取二进制文件: " << file_path << " (元素数: " << count << ")" << std::endl;
}

/**
 * @brief 读取float32格式的矩阵并转换为__fp16
 * @param file_path 二进制文件路径
 * @param data_fp16 输出：__fp16格式矩阵
 * @param count 元素个数
 */
inline void ReadFloat32ToFp16(const std::string &file_path, __fp16 *data_fp16, size_t count)
{
    // 1. 读取float32数据
    float *data_float = new float[count];
    ReadBinFile<float>(file_path, data_float, count);

    // 2. 转换为__fp16
    for (size_t i = 0; i < count; i++)
    {
        data_fp16[i] = (__fp16)data_float[i];
    }

    // 3. 打印前16个元素验证
    std::cout << ">>> 前16个元素（__fp16）:" << std::endl;
    for (size_t i = 0; i < 16 && i < count; i++)
    {
        printf("%.4f ", (float)data_fp16[i]);
        if ((i + 1) % 8 == 0)
            printf("\n");
    }
    printf("\n");

    delete[] data_float;
}

/**
 * @brief 读取float32格式的矩阵
 * @param file_path 二进制文件路径
 * @param data_fp32 输出：float格式矩阵
 * @param count 元素个数
 */
inline void ReadFloat32(const std::string &file_path, float *data_fp32, size_t count)
{
    // 1. 读取float32数据
    ReadBinFile<float>(file_path, data_fp32, count);

    // 2. 打印前16个元素验证
    std::cout << ">>> 前16个元素（float）:" << std::endl;
    for (size_t i = 0; i < 16 && i < count; i++)
    {
        printf("%.4f ", (float)data_fp32[i]);
        if ((i + 1) % 8 == 0)
            printf("\n");
    }
    printf("\n");
}

/**
 * @brief 读取文本元信息文件（仅保留核心功能）
 * @param file_path 元信息文件路径
 * @param key 要读取的键名
 * @return 对应的值
 */
inline int32_t ReadMetaValue(const std::string &file_path, const std::string &key)
{
    std::ifstream meta_file(file_path);
    if (!meta_file.is_open())
    {
        std::cerr << "Error: 无法打开元信息文件 " << file_path << std::endl;
        exit(1);
    }

    std::string line;
    while (std::getline(meta_file, line))
    {
        size_t colon_pos = line.find(":");
        if (colon_pos == std::string::npos)
            continue;

        std::string line_key = line.substr(0, colon_pos);
        line_key.erase(0, line_key.find_first_not_of(" \t"));
        line_key.erase(line_key.find_last_not_of(" \t") + 1);

        if (line_key == key)
        {
            std::string value_str = line.substr(colon_pos + 1);
            value_str.erase(0, value_str.find_first_not_of(" \t"));
            meta_file.close();
            return std::stoi(value_str);
        }
    }

    meta_file.close();
    std::cerr << "Error: 在元信息文件中未找到键 " << key << std::endl;
    exit(1);
}

/**
 * @brief 读取A矩阵的vector-CSR格式（保留核心功能）
 * @param M 矩阵行数
 * @param d 向量块维度
 * @param vec_csr_data 输出：vector-CSR数值（__fp16）
 * @param vec_csr_cols 输出：vector-CSR列索引
 * @param vec_csr_indptr 输出：vector-CSR行指针
 * @param num_vectors 输出：向量块数量
 */
inline void ReadAMatrixVectorCSR(int32_t M, int32_t &d,
                          std::vector<__fp16> &vec_csr_data,
                          std::vector<int32_t> &vec_csr_cols,
                          std::vector<int32_t> &vec_csr_indptr,
                          int32_t &num_vectors)
{
    // 1. 读取vector-CSR元信息
    num_vectors = ReadMetaValue("../../data/vector_csr/A_vector_csr_meta.txt", "num_vector_blocks");
    d = ReadMetaValue("../../data/vector_csr/A_vector_csr_meta.txt", "d");
    int32_t row_blocks = ReadMetaValue("../../data/vector_csr/A_vector_csr_meta.txt", "row_blocks_count");

    // 2. 读取并转换vector-CSR data
    size_t vec_data_count = num_vectors * d;
    float *vec_data_float = new float[vec_data_count];
    ReadBinFile<float>("../../data/vector_csr/A_data.bin", vec_data_float, vec_data_count);

    vec_csr_data.resize(vec_data_count);
    for (size_t i = 0; i < vec_data_count; i++)
    {
        vec_csr_data[i] = (__fp16)vec_data_float[i];
    }

    // 3. 读取vector-CSR cols和indptr
    vec_csr_cols.resize(num_vectors);
    ReadBinFile<int32_t>("../../data/vector_csr/A_cols.bin", vec_csr_cols.data(), num_vectors);

    vec_csr_indptr.resize(row_blocks + 1);
    ReadBinFile<int32_t>("../../data/vector_csr/A_indptr.bin", vec_csr_indptr.data(), row_blocks + 1);

    // 4. 打印验证信息
    std::cout << "\n>>> A矩阵vector-CSR格式信息:" << std::endl;
    std::cout << "向量块数量: " << num_vectors << "  向量维度d: " << d << std::endl;
    std::cout << "行块数: " << row_blocks << "  行指针长度: " << vec_csr_indptr.size() << std::endl;
    std::cout << ">>> 前16个元素（float）:" << std::endl;
    for (size_t i = 0; i < 16 && i < vec_data_count; i++)
    {
        printf("%.4f ", (float)vec_data_float[i]);
        if ((i + 1) % 8 == 0)
            printf("\n");
    }
    printf("\n");
    delete[] vec_data_float;
}

#endif // FILE_UTILS_H