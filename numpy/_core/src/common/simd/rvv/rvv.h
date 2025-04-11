#ifndef _NPY_SIMD_H_
    #error "Not a standalone header"
#endif

#include <riscv_vector.h>

// supports VLEN 128, 256 and 512
// it is impossible to implement npyv_tobits_b8 when VLEN>512
#define NPY_SIMD __riscv_v_fixed_vlen
#define NPY_SIMD_WIDTH (__riscv_v_fixed_vlen / 8)
#define NPY_SIMD_F32 1
#define NPY_SIMD_F64 1

#ifdef NPY_HAVE_FMA3
    #define NPY_SIMD_FMA3 1 // native support
#else
    #define NPY_SIMD_FMA3 0 // fast emulated
#endif

#define NPY_SIMD_BIGENDIAN 0
#define NPY_SIMD_CMPSIGNAL 1

typedef vuint8m1_t fixed_vuint8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint16m1_t fixed_vuint16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint32m1_t fixed_vuint32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vuint64m1_t fixed_vuint64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint8m1_t fixed_vint8m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint16m1_t fixed_vint16m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint32m1_t fixed_vint32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vint64m1_t fixed_vint64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vfloat32m1_t fixed_vfloat32m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));
typedef vfloat64m1_t fixed_vfloat64m1_t __attribute__((riscv_rvv_vector_bits(__riscv_v_fixed_vlen)));

#define npyv_u8 fixed_vuint8m1_t
#define npyv_u16 fixed_vuint16m1_t
#define npyv_u32 fixed_vuint32m1_t
#define npyv_u64 fixed_vuint64m1_t
#define npyv_s8 fixed_vint8m1_t
#define npyv_s16 fixed_vint16m1_t
#define npyv_s32 fixed_vint32m1_t
#define npyv_s64 fixed_vint64m1_t
#define npyv_f32 fixed_vfloat32m1_t
#define npyv_f64 fixed_vfloat64m1_t

// simulate bool as uint due to gcc/clang bugs, change to fixed_vbool if possible
#define npyv_b8 fixed_vuint8m1_t
#define npyv_b16 fixed_vuint16m1_t
#define npyv_b32 fixed_vuint32m1_t
#define npyv_b64 fixed_vuint64m1_t


typedef struct { fixed_vuint8m1_t val[2]; } npyv_u8x2;
typedef struct { fixed_vint8m1_t val[2]; } npyv_s8x2;
typedef struct { fixed_vuint16m1_t val[2]; } npyv_u16x2;
typedef struct { fixed_vint16m1_t val[2]; } npyv_s16x2;
typedef struct { fixed_vuint32m1_t val[2]; } npyv_u32x2;
typedef struct { fixed_vint32m1_t val[2]; } npyv_s32x2;
typedef struct { fixed_vuint64m1_t val[2]; } npyv_u64x2;
typedef struct { fixed_vint64m1_t val[2]; } npyv_s64x2;
typedef struct { fixed_vfloat32m1_t val[2]; } npyv_f32x2;
typedef struct { fixed_vfloat64m1_t val[2]; } npyv_f64x2;


typedef struct { fixed_vuint8m1_t val[3]; } npyv_u8x3;
typedef struct { fixed_vint8m1_t val[3]; } npyv_s8x3;
typedef struct { fixed_vuint16m1_t val[3]; } npyv_u16x3;
typedef struct { fixed_vint16m1_t val[3]; } npyv_s16x3;
typedef struct { fixed_vuint32m1_t val[3]; } npyv_u32x3;
typedef struct { fixed_vint32m1_t val[3]; } npyv_s32x3;
typedef struct { fixed_vuint64m1_t val[3]; } npyv_u64x3;
typedef struct { fixed_vint64m1_t val[3]; } npyv_s64x3;
typedef struct { fixed_vfloat32m1_t val[3]; } npyv_f32x3;
typedef struct { fixed_vfloat64m1_t val[3]; } npyv_f64x3;


// helper types
#define npyv__u8x2 vuint8m1x2_t
#define npyv__u16x2 vuint16m1x2_t
#define npyv__u32x2 vuint32m1x2_t
#define npyv__u64x2 vuint64m1x2_t
#define npyv__s8x2 vint8m1x2_t
#define npyv__s16x2 vint16m1x2_t
#define npyv__s32x2 vint32m1x2_t
#define npyv__s64x2 vint64m1x2_t
#define npyv__f32x2 vfloat32m1x2_t
#define npyv__f64x2 vfloat64m1x2_t


#define npyv_nlanes_u8  32
#define npyv_nlanes_s8  32
#define npyv_nlanes_u16 16
#define npyv_nlanes_s16 16
#define npyv_nlanes_u32 8
#define npyv_nlanes_s32 8
#define npyv_nlanes_u64 4
#define npyv_nlanes_s64 4
#define npyv_nlanes_f32 8
#define npyv_nlanes_f64 4

#include "memory.h"
#include "misc.h"
#include "reorder.h"
#include "operators.h"
#include "conversion.h"
#include "arithmetic.h"
#include "math.h"
