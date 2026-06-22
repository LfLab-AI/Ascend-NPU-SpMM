#ifndef DATA_UTILS_H
#define DATA_UTILS_H
#include <iostream>
#include <fstream>
#include <cstdio>
#include <string>
#include <vector>
#include <iomanip>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include "acl/acl.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN] " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)
#define RETURN_IF_NOT_SUCCESS(expr, format, action)  \
    do {                             \
        if ((expr) != ACL_SUCCESS) { \
            ERROR_LOG(format);       \
            action;                  \
        }                            \
    } while (0)

#define CALL_RT(x) \
    if (auto ret = (x) != 0) { \
        std::cout << "[ERROR] Failed to exec acl api " << #x << ", result: " << ret << std::endl; \
        return -1; \
    } else { \
        /*\
        // 正确不需要输出信息\
        std::cout << "[INFO] Succeeded to exec acl api " << #x << std::endl; */\
    }


#define IF_NOT_SUCCESS_RETURN_FALSE(expr, format, action)  \
    do {                             \
        if ((expr) != ACL_SUCCESS) { \
            ERROR_LOG(format);       \
            action;                  \
            return false;            \
        }                            \
    } while (0)

#define ACL_ERROR_LOG(fmt, args...) fprintf(stdout, "[ACL_ERROR]  " fmt "\n", ##args)


/**
 * @brief Get Acl Recent Error Message.
 */
void GetRecentErrMsg()
{
    const char *aclRecentErrMsg = nullptr;
    aclRecentErrMsg = aclGetRecentErrMsg();
    if (aclRecentErrMsg != nullptr) {
        ACL_ERROR_LOG("%s", aclRecentErrMsg);
    } else {
        ACL_ERROR_LOG("Failed to get recent error message.");
    }
}

/**
 * @brief Read data from file
 * @param [in] filePath: file path
 * @param [out] fileSize: file size
 * @return read result
 */
bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("%s Failed to get file.", filePath.c_str());
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("file size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("file size is larger than buffer size");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

/**
 * @brief Write data to file
 * @param [in] filePath: file path
 * @param [in] buffer: data to write to file
 * @param [in] size: size to write
 * @return write result
 */
bool WriteFile(const std::string &filePath, const void *buffer, size_t size)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. buffer is nullptr");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
    if (fd < 0) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    auto writeSize = write(fd, buffer, size);
    (void) close(fd);
    if (writeSize != size) {
        ERROR_LOG("Write file Failed.");
        return false;
    }

    return true;
}
#endif // DATA_UTILS_H

/**
 * @brief compare actual data and expected data
 * @param [in] T *actualOutputData: actual data
 * @param [in] T *expectedOutputData: expected data
 * @param [in] uint64_t len: data length
 * @return isCorrect
 */
template<typename T>
bool compareOutputData(T *actualOutputData, T *expectedOutputData, uint64_t len)
{
    float thres;
    if (std::is_same<T, float>::value) {
        thres = 1e-4;
    } else if (std::is_same<T, __fp16>::value) {
        thres = 1e-3;
    }
    double error = 0;
    int64_t errorCount = 0;
    int64_t errorOutputCount = 0;
    int64_t lastError = -1;
    int64_t continuous = 0;
    int64_t maxContinous = 0;
    int64_t i = 0;
    float ratios[] = {thres, thres};
    for (i = 0; i < len; i++) {
        T actualOutputItem = *(actualOutputData + i);
        T expectedOutputItem = *(expectedOutputData + i);
        if (i >= 0 && i < 10) {
            printf("our: %f, expected: %f\n", actualOutputItem, expectedOutputItem);
        }
        float limitError = std::abs((std::min(expectedOutputItem, actualOutputItem)) * ratios[1]);
        if (std::abs(actualOutputItem - expectedOutputItem) > 0.1 && errorOutputCount < 16) {
            ++errorOutputCount;
            std::cout << "index: " << i <<  " sub super 0.1! actual:" << actualOutputItem <<  ", expected:" << expectedOutputItem << std::endl;
        }
        if (std::abs(actualOutputItem - expectedOutputItem) > limitError) {
            errorCount++;
            if (i == lastError + 1) {
                continuous++;
            } else {
                if (maxContinous < continuous) {
                    maxContinous = continuous;
                }
                continuous = 1;
            }
            lastError = i;
        }
        error += std::min(1.0f, std::abs((actualOutputItem - expectedOutputItem) / std::min(expectedOutputItem, actualOutputItem)));
    }
    error = error / len;
    std::cout << "error: " << error << std::endl;
    if (i == len - 1) {
        if (maxContinous < continuous) {
            maxContinous = continuous;
        }
    }

    int count = 0;
    if (errorCount >= len * ratios[0] || maxContinous > 16) {
        for (i = 0; i < len; i++) {          
            T actualOutputItem = *(actualOutputData + i);
            T expectedOutputItem = *(expectedOutputData + i);
            float limitError = std::abs((std::min(expectedOutputItem, actualOutputItem)) * ratios[1]);
            if (std::abs(actualOutputItem - expectedOutputItem) > limitError && errorCount < 16) {
                std::cout << "index:" << i << " ,cmprlst:" << std::abs((actualOutputItem - expectedOutputItem) / std::min(expectedOutputItem, actualOutputItem)) <<
                    " ,actualDataf:" << actualOutputItem << " ,expectedDataf:" <<
                    expectedOutputItem << std::endl;
            }
        }
        printf("cmp len:%lu\n",len);
        std::cout << "errorCount:" << errorCount << std::endl;
        printf("ratio:%.10lf\n",(double)errorCount / len);
        return false;
    } else {
        printf("cmp len:%lu\n",len);
        std::cout << "errorCount:" << errorCount << std::endl;
        printf("ratio:%.10lf\n",(double)errorCount / len);
        return true;
    }
}

// 输出错误指标：最大绝对误差，最大相对误差，平均相对误差，平均绝对误差，相对误差超过0.0001的个数和比例
template<typename T>
void outputError(T *actualOutputData, T *expectedOutputData, uint64_t len)
{
    float thres;
    if (std::is_same<T, float>::value) {
        thres = 1e-4;
    } else if (std::is_same<T, __fp16>::value) {
        thres = 1e-3;
    }
    double max_abs_error = 0.0f;   // 最大绝对误差
    double max_rel_error = 0.0f;   // 最大相对误差
    double sum_abs_error = 0.0f;   // 累积的绝对误差
    double sum_rel_error = 0.0f;   // 累计的相对误差
    double avg_abs_error = 0.0f;   // 平均绝对误差
    double avg_rel_error = 0.0f;   // 平均相对误差
    int64_t errorCount = 0;        // 相对误差超过0.0001的个数
    double errorCountRatio = 0.0f; // 相对误差超过0.0001的比例

    uint64_t i = 0;
    for (i = 0; i < len; i++)
    {
        T actualOutputItem = *(actualOutputData + i);
        T expectedOutputItem = *(expectedOutputData + i);
        float cur_abs_error = std::abs(actualOutputItem - expectedOutputItem);
        float cur_rel_error = cur_abs_error / (std::max(expectedOutputItem, actualOutputItem) + 1e-7);
        // 更新最大绝对误差
        if (cur_abs_error > max_abs_error)
        {
            max_abs_error = cur_abs_error;
        }
        // 更新最大相对误差
        if (cur_rel_error > max_rel_error)
        {
            max_rel_error = cur_rel_error;
        }
        // 累计相对误差和绝对误差
        sum_rel_error += std::min(1.0f, cur_rel_error);
        sum_abs_error += std::min(1.0f, cur_abs_error);
        if (cur_rel_error > thres)
        {
            errorCount++;
        }
    }
    // 平均相对误差和平均绝对误差
    avg_rel_error = sum_rel_error / len;
    avg_abs_error = sum_abs_error / len;
    errorCountRatio = 1.0 * errorCount / len;
    printf("error_static: %.2e, %.2e, %.2e, %.2e, %ld, %.2e\n", max_abs_error, max_rel_error, avg_abs_error, avg_rel_error, errorCount, errorCountRatio);
}


ascblasOperation_t num2Op(int transNum) {
    ascblasOperation_t trans;
    switch (transNum)
    {
    case 1:
        trans = ASCBLAS_OP_T;
        break;
    case 2:
        trans = ASCBLAS_OP_C;
        break;
    default:
        trans = ASCBLAS_OP_N;
        break;
    }
    return trans;
}