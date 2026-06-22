#ifndef ASCBLAS_TYPE
#define ASCBLAS_TYPE

#ifdef ASCEND910B1
#define CORENUM 20
#elif ASCEND910B2
#define CORENUM 30
#elif ASCEND910B3
#define CORENUM 20
#else
#define CORENUM 20
#endif

typedef enum
{
    ASCBLAS_SIDE_LEFT,
    ASCBLAS_SIDE_RIGHT
} ascblasSideMode_t;

typedef enum
{
    ASCBLAS_OP_N,
    ASCBLAS_OP_T,
    ASCBLAS_OP_C
} ascblasOperation_t;

typedef enum
{
    ASCBLAS_FILL_MODE_LOWER,
    ASCBLAS_FILL_MODE_UPPER,
    ASCBLAS_FILL_MODE_FULL
} ascblasFillMode_t;

typedef enum
{
    ASCBLAS_DIAG_NON_UNIT,
    ASCBLAS_DIAG_UNIT
} ascblasDiagType_t;

typedef struct {
    float real;
    float imag;
} ascComplex;

#endif
