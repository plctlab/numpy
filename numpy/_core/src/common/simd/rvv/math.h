#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_MATH_H
#define _NPY_SIMD_RVV_MATH_H

#include <fenv.h>
#include <float.h>

/***************************
 * Elementary
 ***************************/
NPY_FINLINE npyv_f32 npyv_abs_f32(npyv_f32 a)
{ return __riscv_vfabs_v_f32m1(a, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_abs_f64(npyv_f64 a)
{ return __riscv_vfabs_v_f64m1(a, npyv_nlanes_f64); }

// Square
NPY_FINLINE npyv_f32 npyv_square_f32(npyv_f32 a)
{ return __riscv_vfmul_vv_f32m1(a, a, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_square_f64(npyv_f64 a)
{ return __riscv_vfmul_vv_f64m1(a, a, npyv_nlanes_f64); }

// Square root
NPY_FINLINE npyv_f32 npyv_sqrt_f32(npyv_f32 a)
{ return __riscv_vfsqrt_v_f32m1(a, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_sqrt_f64(npyv_f64 a)
{ return __riscv_vfsqrt_v_f64m1(a, npyv_nlanes_f64); }

// Reciprocal
NPY_FINLINE npyv_f32 npyv_recip_f32(npyv_f32 a)
{ return __riscv_vfrdiv_vf_f32m1(a, 1.0f, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_recip_f64(npyv_f64 a)
{ return __riscv_vfrdiv_vf_f64m1(a, 1.0 , npyv_nlanes_f64); }

// Maximum
NPY_FINLINE npyv_f32 npyv_max_f32(npyv_f32 a, npyv_f32 b)
{ return __riscv_vfmax_vv_f32m1(a, b, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_max_f64(npyv_f64 a, npyv_f64 b)
{ return __riscv_vfmax_vv_f64m1(a, b, npyv_nlanes_f64); }

// Max, NaN-suppressing
#define npyv_maxp_f32 npyv_max_f32
#define npyv_maxp_f64 npyv_max_f64

// Max, NaN-propagating
NPY_FINLINE npyv_f32 npyv_maxn_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vfmax_vv_f32m1(
        __riscv_vmerge(b, a, __riscv_vmfeq(b, b, npyv_nlanes_f32), npyv_nlanes_f32),
        __riscv_vmerge(a, b, __riscv_vmfeq(a, a, npyv_nlanes_f32), npyv_nlanes_f32),
        npyv_nlanes_f32
    );
}
NPY_FINLINE npyv_f64 npyv_maxn_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vfmax_vv_f64m1(
        __riscv_vmerge(b, a, __riscv_vmfeq(b, b, npyv_nlanes_f64), npyv_nlanes_f64),
        __riscv_vmerge(a, b, __riscv_vmfeq(a, a, npyv_nlanes_f64), npyv_nlanes_f64),
        npyv_nlanes_f64
    );
}

// Maximum, integer operations
NPY_FINLINE npyv_u8 npyv_max_u8(npyv_u8 a, npyv_u8 b)
{ return __riscv_vmaxu_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_max_s8(npyv_s8 a, npyv_s8 b)
{ return __riscv_vmax_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_max_u16(npyv_u16 a, npyv_u16 b)
{ return __riscv_vmaxu_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_max_s16(npyv_s16 a, npyv_s16 b)
{ return __riscv_vmax_vv_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_max_u32(npyv_u32 a, npyv_u32 b)
{ return __riscv_vmaxu_vv_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_max_s32(npyv_s32 a, npyv_s32 b)
{ return __riscv_vmax_vv_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_max_u64(npyv_u64 a, npyv_u64 b)
{ return __riscv_vmaxu_vv_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_max_s64(npyv_s64 a, npyv_s64 b)
{ return __riscv_vmax_vv_i64m1(a, b, npyv_nlanes_s64); }

// Minimum
NPY_FINLINE npyv_f32 npyv_min_f32(npyv_f32 a, npyv_f32 b)
{ return __riscv_vfmin_vv_f32m1(a, b, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_min_f64(npyv_f64 a, npyv_f64 b)
{ return __riscv_vfmin_vv_f64m1(a, b, npyv_nlanes_f64); }

// Min, NaN-suppressing
#define npyv_minp_f32 npyv_min_f32
#define npyv_minp_f64 npyv_min_f64

// Min, NaN-propagating
NPY_FINLINE npyv_f32 npyv_minn_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vfmin_vv_f32m1(
        __riscv_vmerge(b, a, __riscv_vmfeq(b, b, npyv_nlanes_f32), npyv_nlanes_f32),
        __riscv_vmerge(a, b, __riscv_vmfeq(a, a, npyv_nlanes_f32), npyv_nlanes_f32),
        npyv_nlanes_f32
    );
}
NPY_FINLINE npyv_f64 npyv_minn_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vfmin_vv_f64m1(
        __riscv_vmerge(b, a, __riscv_vmfeq(b, b, npyv_nlanes_f64), npyv_nlanes_f64),
        __riscv_vmerge(a, b, __riscv_vmfeq(a, a, npyv_nlanes_f64), npyv_nlanes_f64),
        npyv_nlanes_f64
    );
}

// Minimum, integer operations
NPY_FINLINE npyv_u8 npyv_min_u8(npyv_u8 a, npyv_u8 b)
{ return __riscv_vminu_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_min_s8(npyv_s8 a, npyv_s8 b)
{ return __riscv_vmin_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_min_u16(npyv_u16 a, npyv_u16 b)
{ return __riscv_vminu_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_min_s16(npyv_s16 a, npyv_s16 b)
{ return __riscv_vmin_vv_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_min_u32(npyv_u32 a, npyv_u32 b)
{ return __riscv_vminu_vv_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_min_s32(npyv_s32 a, npyv_s32 b)
{ return __riscv_vmin_vv_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_min_u64(npyv_u64 a, npyv_u64 b)
{ return __riscv_vminu_vv_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_min_s64(npyv_s64 a, npyv_s64 b)
{ return __riscv_vmin_vv_i64m1(a, b, npyv_nlanes_s64); }

// reduce min/max for all data types
// Maximum reductions
NPY_FINLINE uint8_t npyv_reduce_max_u8(npyv_u8 a)
{ return __riscv_vmv_x_s_u8m1_u8(__riscv_vredmaxu_vs_u8m1_u8m1(a, __riscv_vmv_s_x_u8m1(0, 1), npyv_nlanes_u8)); }
NPY_FINLINE int8_t npyv_reduce_max_s8(npyv_s8 a)
{ return __riscv_vmv_x_s_i8m1_i8(__riscv_vredmax_vs_i8m1_i8m1(a, __riscv_vmv_s_x_i8m1(INT8_MIN, 1), npyv_nlanes_s8)); }
NPY_FINLINE uint16_t npyv_reduce_max_u16(npyv_u16 a)
{ return __riscv_vmv_x_s_u16m1_u16(__riscv_vredmaxu_vs_u16m1_u16m1(a, __riscv_vmv_s_x_u16m1(0, 1), npyv_nlanes_u16)); }
NPY_FINLINE int16_t npyv_reduce_max_s16(npyv_s16 a)
{ return __riscv_vmv_x_s_i16m1_i16(__riscv_vredmax_vs_i16m1_i16m1(a, __riscv_vmv_s_x_i16m1(INT16_MIN, 1), npyv_nlanes_s16)); }
NPY_FINLINE uint32_t npyv_reduce_max_u32(npyv_u32 a)
{ return __riscv_vmv_x_s_u32m1_u32(__riscv_vredmaxu_vs_u32m1_u32m1(a, __riscv_vmv_s_x_u32m1(0, 1), npyv_nlanes_u32)); }
NPY_FINLINE int32_t npyv_reduce_max_s32(npyv_s32 a)
{ return __riscv_vmv_x_s_i32m1_i32(__riscv_vredmax_vs_i32m1_i32m1(a, __riscv_vmv_s_x_i32m1(INT32_MIN, 1), npyv_nlanes_s32)); }
NPY_FINLINE uint64_t npyv_reduce_max_u64(npyv_u64 a)
{ return __riscv_vmv_x_s_u64m1_u64(__riscv_vredmaxu_vs_u64m1_u64m1(a, __riscv_vmv_s_x_u64m1(0, 1), npyv_nlanes_u64)); }
NPY_FINLINE int64_t npyv_reduce_max_s64(npyv_s64 a)
{ return __riscv_vmv_x_s_i64m1_i64(__riscv_vredmax_vs_i64m1_i64m1(a, __riscv_vmv_s_x_i64m1(INT64_MIN, 1), npyv_nlanes_s64)); }

// Floating-point maximum reductions
NPY_FINLINE float npyv_reduce_max_f32(npyv_f32 a)
{ return __riscv_vfirst(__riscv_vmfeq(a, a, npyv_nlanes_f32), npyv_nlanes_f32) != -1 ? __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmax_vs_f32m1_f32m1(a, __riscv_vfmv_s_f_f32m1(-INFINITY, 1), npyv_nlanes_f32)) : NAN; }
NPY_FINLINE double npyv_reduce_max_f64(npyv_f64 a)
{ return __riscv_vfirst(__riscv_vmfeq(a, a, npyv_nlanes_f64), npyv_nlanes_f64) != -1 ? __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmax_vs_f64m1_f64m1(a, __riscv_vfmv_s_f_f64m1(-INFINITY, 1), npyv_nlanes_f64)) : NAN; }

// NaN-suppressing maximum reductions
#define npyv_reduce_maxp_f32 npyv_reduce_max_f32
#define npyv_reduce_maxp_f64 npyv_reduce_max_f64

// NaN-propagating maximum reductions
NPY_FINLINE float npyv_reduce_maxn_f32(npyv_f32 a)
{ return __riscv_vfirst(__riscv_vmfne(a, a, npyv_nlanes_f32), npyv_nlanes_f32) == -1 ? npyv_reduce_max_f32(a) : NAN; }
NPY_FINLINE double npyv_reduce_maxn_f64(npyv_f64 a)
{ return __riscv_vfirst(__riscv_vmfne(a, a, npyv_nlanes_f64), npyv_nlanes_f64) == -1 ? npyv_reduce_max_f64(a) : NAN; }

// Minimum reductions
NPY_FINLINE uint8_t npyv_reduce_min_u8(npyv_u8 a)
{ return __riscv_vmv_x_s_u8m1_u8(__riscv_vredminu_vs_u8m1_u8m1(a, __riscv_vmv_s_x_u8m1(UINT8_MAX, 1), npyv_nlanes_u8)); }
NPY_FINLINE int8_t npyv_reduce_min_s8(npyv_s8 a)
{ return __riscv_vmv_x_s_i8m1_i8(__riscv_vredmin_vs_i8m1_i8m1(a, __riscv_vmv_s_x_i8m1(INT8_MAX, 1), npyv_nlanes_s8)); }
NPY_FINLINE uint16_t npyv_reduce_min_u16(npyv_u16 a)
{ return __riscv_vmv_x_s_u16m1_u16(__riscv_vredminu_vs_u16m1_u16m1(a, __riscv_vmv_s_x_u16m1(UINT16_MAX, 1), npyv_nlanes_u16)); }
NPY_FINLINE int16_t npyv_reduce_min_s16(npyv_s16 a)
{ return __riscv_vmv_x_s_i16m1_i16(__riscv_vredmin_vs_i16m1_i16m1(a, __riscv_vmv_s_x_i16m1(INT16_MAX, 1), npyv_nlanes_s16)); }
NPY_FINLINE uint32_t npyv_reduce_min_u32(npyv_u32 a)
{ return __riscv_vmv_x_s_u32m1_u32(__riscv_vredminu_vs_u32m1_u32m1(a, __riscv_vmv_s_x_u32m1(UINT32_MAX, 1), npyv_nlanes_u32)); }
NPY_FINLINE int32_t npyv_reduce_min_s32(npyv_s32 a)
{ return __riscv_vmv_x_s_i32m1_i32(__riscv_vredmin_vs_i32m1_i32m1(a, __riscv_vmv_s_x_i32m1(INT32_MAX, 1), npyv_nlanes_s32)); }
NPY_FINLINE uint64_t npyv_reduce_min_u64(npyv_u64 a)
{ return __riscv_vmv_x_s_u64m1_u64(__riscv_vredminu_vs_u64m1_u64m1(a, __riscv_vmv_s_x_u64m1(UINT64_MAX, 1), npyv_nlanes_u64)); }
NPY_FINLINE int64_t npyv_reduce_min_s64(npyv_s64 a)
{ return __riscv_vmv_x_s_i64m1_i64(__riscv_vredmin_vs_i64m1_i64m1(a, __riscv_vmv_s_x_i64m1(INT64_MAX, 1), npyv_nlanes_s64)); }

// Floating-point minimum reductions
NPY_FINLINE float npyv_reduce_min_f32(npyv_f32 a)
{ return __riscv_vfirst(__riscv_vmfeq(a, a, npyv_nlanes_f32), npyv_nlanes_f32) != -1 ? __riscv_vfmv_f_s_f32m1_f32(__riscv_vfredmin_vs_f32m1_f32m1(a, __riscv_vfmv_s_f_f32m1(INFINITY, 1), npyv_nlanes_f32)) : NAN; }
NPY_FINLINE double npyv_reduce_min_f64(npyv_f64 a)
{ return __riscv_vfirst(__riscv_vmfeq(a, a, npyv_nlanes_f64), npyv_nlanes_f64) != -1 ? __riscv_vfmv_f_s_f64m1_f64(__riscv_vfredmin_vs_f64m1_f64m1(a, __riscv_vfmv_s_f_f64m1(INFINITY, 1), npyv_nlanes_f64)) : NAN; }

// NaN-suppressing minimum reductions
#define npyv_reduce_minp_f32 npyv_reduce_min_f32
#define npyv_reduce_minp_f64 npyv_reduce_min_f64

// NaN-propagating minimum reductions
NPY_FINLINE float npyv_reduce_minn_f32(npyv_f32 a)
{ return __riscv_vfirst(__riscv_vmfne(a, a, npyv_nlanes_f32), npyv_nlanes_f32) == -1 ? npyv_reduce_min_f32(a) : NAN; }
NPY_FINLINE double npyv_reduce_minn_f64(npyv_f64 a)
{ return __riscv_vfirst(__riscv_vmfne(a, a, npyv_nlanes_f64), npyv_nlanes_f64) == -1 ? npyv_reduce_min_f64(a) : NAN; }

#define NPYV_IMPL_RVV_FCVT(TYPE, FRM)                     \
    NPY_FINLINE npyv_f32 npyv_##TYPE##_f32(npyv_f32 a)    \
    {                                                     \
        const int vl = npyv_nlanes_f32;                   \
        const vfloat32m1_t b = __riscv_vmerge(            \
            a,                                            \
            __riscv_vfcvt_f(__riscv_vfcvt_x_f_v_i32m1_rm( \
                a, FRM, vl), vl                           \
            ),                                            \
            __riscv_vmfle(                                \
                __riscv_vfabs(a, vl), 1e9, vl             \
            ), vl                                         \
        );                                                \
        feclearexcept(FE_INVALID);                        \
        return __riscv_vreinterpret_f32m1(__riscv_vor(    \
            __riscv_vand(                                 \
                __riscv_vreinterpret_u32m1(a),            \
                1 << 31, vl                               \
            ),                                            \
            __riscv_vreinterpret_u32m1(b), vl             \
        ));                                               \
    }                                                     \
    NPY_FINLINE npyv_f64 npyv_##TYPE##_f64(npyv_f64 a)    \
    {                                                     \
        const int vl = npyv_nlanes_f64;                   \
        const vfloat64m1_t b = __riscv_vmerge(            \
            a,                                            \
            __riscv_vfcvt_f(__riscv_vfcvt_x_f_v_i64m1_rm( \
                a, FRM, vl), vl                           \
            ),                                            \
            __riscv_vmfle(                                \
                __riscv_vfabs(a, vl), 1e18, vl            \
            ), vl                                         \
        );                                                \
        feclearexcept(FE_INVALID);                        \
        return __riscv_vreinterpret_f64m1(__riscv_vor(    \
            __riscv_vand(                                 \
                __riscv_vreinterpret_u64m1(a),            \
                1ULL << 63, vl                            \
            ),                                            \
            __riscv_vreinterpret_u64m1(b), vl             \
        ));                                               \
    }

// round to nearest integer even
NPYV_IMPL_RVV_FCVT(rint, __RISCV_FRM_RNE)
// trunc
NPYV_IMPL_RVV_FCVT(trunc, __RISCV_FRM_RTZ)
// ceil
NPYV_IMPL_RVV_FCVT(ceil, __RISCV_FRM_RUP)
// floor
NPYV_IMPL_RVV_FCVT(floor, __RISCV_FRM_RDN)
#undef NPYV_IMPL_RVV_FCVT

#endif // _NPY_SIMD_RVV_MATH_H
