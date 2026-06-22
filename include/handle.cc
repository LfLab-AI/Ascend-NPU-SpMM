#include "handle.h"

aclError ascblasCreate(ascblasHandle_t *handle)
{
    aclError error = aclrtCreateStream(&(handle->stream));
    return error;
}

aclError ascblasDestroy(ascblasHandle_t handle)
{
    aclError error = aclrtDestroyStream(handle.stream);
    return error;
}

void ascblasSetStream(ascblasHandle_t handle, aclrtStream streamId)
{
    handle.stream = streamId;
}

void ascblasGetStream(ascblasHandle_t handle, aclrtStream *streamId)
{
    *streamId = handle.stream;
}