#ifndef ASCBLAS_H
#define ASCBLAS_H

#include "rt.h"
#include "ascblas_type.h"
#include "handle.h"

char* ReadBinFile(const char* fileName, uint32_t* fileSize)
{
    std::filebuf* fBuf;
    std::ifstream fileStream;
    size_t size;

    fileStream.open(fileName, std::ios::binary);
    if (!fileStream) {
        printf("Cannot open file %s\n", fileName);
        exit(1);
    }

    fBuf = fileStream.rdbuf();
    size = fBuf->pubseekoff(0, std::ios::end, std::ios::in);
    fBuf->pubseekpos(0, std::ios::in);
    char* buffer = new char[size];
    if (buffer == nullptr) {
        printf("New failed\n");
        exit(1);
    }

    fBuf->sgetn(buffer, size);

    fileStream.close();
    *fileSize = size;
    return buffer;
}

void RegisterBinaryKernel(const char* funcName, const char* binFile)
{
    char* buffer = nullptr;
    rtDevBinary_t binary;
    void* binHandle = nullptr;

    uint32_t bufferSize = 0;
    buffer = ReadBinFile(binFile, &bufferSize);
    if (buffer == nullptr) {
        printf("readBinFile failed\n");
        exit(1);
    }

    binary.data = buffer;
    binary.length = bufferSize;
    binary.magic = RT_DEV_BINARY_MAGIC_ELF;
    binary.version = 0;

    aclError rtRet = rtDevBinaryRegister(&binary, &binHandle);
    if (rtRet != ACL_ERROR_NONE) {
        printf("rtDevBinaryRegister failed: %d\n", rtRet);
        // exit(1);
    }

    rtRet = rtFunctionRegister(binHandle, funcName, funcName, (void*)funcName, 0);
    if (rtRet != ACL_ERROR_NONE) {
        printf("rtFunctionRegister failed: %d\n", rtRet);
        // exit(1);
    }
}


#endif
