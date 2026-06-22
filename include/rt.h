#ifndef __MIX_KERNEL_DEMO_RT_H__
#define __MIX_KERNEL_DEMO_RT_H__
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


#ifdef __cplusplus
extern "C" {
#endif

typedef void* aclrtStream;

#define RT_DEV_BINARY_MAGIC_ELF 0x43554245

typedef struct tagRtDevBinary {
    uint32_t magic;
    uint32_t version;
    const void* data;
    uint64_t length;
} rtDevBinary_t;

aclError rtDevBinaryRegister(
    const rtDevBinary_t* bin,
    void** handle
);

aclError rtKernelLaunch(
    const void* stubFunc,
    uint32_t blockDim,
    void* args,
    uint32_t argsSize,
    void* smDesc,
    aclrtStream stream
);

aclError rtFunctionRegister(
    void* binHandle,
    const void* stubFunc,
    const char* stubName,
    const void* devFunc,
    uint32_t funcMode
);

aclError rtGetC2cCtrlAddr(uint64_t* addr, uint32_t* len);

#ifdef __cplusplus
}
#endif

#endif // __MIX_KERNEL_DEMO_RT_H__
