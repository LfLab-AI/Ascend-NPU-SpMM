#ifndef HANDLE_H
#define HANDLE_H
#include "rt.h"
#include "acl/acl.h"
struct ascblasHandle_t
{
private:
    aclrtStream stream;

public:
    friend aclError ascblasCreate(ascblasHandle_t *handle);
    friend aclError ascblasDestroy(ascblasHandle_t handle);
    friend void ascblasSetStream(ascblasHandle_t handle, aclrtStream streamId);
    friend void ascblasGetStream(ascblasHandle_t handle, aclrtStream *streamId);
};
#endif