#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_REORDER_H
#define _NPY_SIMD_RVV_REORDER_H

// combine lower part of two vectors
NPY_FINLINE npyv_u8 npyv_combinel_u8(npyv_u8 a, npyv_u8 b)
{ return __riscv_vslideup_vx_u8m1(a, b, npyv_nlanes_u8 / 2, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_combinel_s8(npyv_s8 a, npyv_s8 b)
{ return __riscv_vslideup_vx_i8m1(a, b, npyv_nlanes_s8 / 2, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_combinel_u16(npyv_u16 a, npyv_u16 b)
{ return __riscv_vslideup_vx_u16m1(a, b, npyv_nlanes_u16 / 2, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_combinel_s16(npyv_s16 a, npyv_s16 b)
{ return __riscv_vslideup_vx_i16m1(a, b, npyv_nlanes_s16 / 2, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_combinel_u32(npyv_u32 a, npyv_u32 b)
{ return __riscv_vslideup_vx_u32m1(a, b, npyv_nlanes_u32 / 2, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_combinel_s32(npyv_s32 a, npyv_s32 b)
{ return __riscv_vslideup_vx_i32m1(a, b, npyv_nlanes_s32 / 2, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_combinel_u64(npyv_u64 a, npyv_u64 b)
{ return __riscv_vslideup_vx_u64m1(a, b, npyv_nlanes_u64 / 2, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_combinel_s64(npyv_s64 a, npyv_s64 b)
{ return __riscv_vslideup_vx_i64m1(a, b, npyv_nlanes_s64 / 2, npyv_nlanes_s64); }
NPY_FINLINE npyv_f32 npyv_combinel_f32(npyv_f32 a, npyv_f32 b)
{ return __riscv_vslideup_vx_f32m1(a, b, npyv_nlanes_f32 / 2, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_combinel_f64(npyv_f64 a, npyv_f64 b)
{ return __riscv_vslideup_vx_f64m1(a, b, npyv_nlanes_f64 / 2, npyv_nlanes_f64); }

// combine higher part of two vectors
NPY_FINLINE npyv_u8 npyv_combineh_u8(npyv_u8 a, npyv_u8 b)
{
    return __riscv_vslideup_vx_u8m1(
        __riscv_vslidedown_vx_u8m1(a, npyv_nlanes_u8 / 2, npyv_nlanes_u8),
        __riscv_vslidedown_vx_u8m1(b, npyv_nlanes_u8 / 2, npyv_nlanes_u8),
        npyv_nlanes_u8 / 2,
        npyv_nlanes_u8
    );
}

NPY_FINLINE npyv_u16 npyv_combineh_u16(npyv_u16 a, npyv_u16 b)
{
    return __riscv_vslideup_vx_u16m1(
        __riscv_vslidedown_vx_u16m1(a, npyv_nlanes_u16 / 2, npyv_nlanes_u16),
        __riscv_vslidedown_vx_u16m1(b, npyv_nlanes_u16 / 2, npyv_nlanes_u16),
        npyv_nlanes_u16 / 2,
        npyv_nlanes_u16
    );
}

NPY_FINLINE npyv_u32 npyv_combineh_u32(npyv_u32 a, npyv_u32 b)
{
    return __riscv_vslideup_vx_u32m1(
        __riscv_vslidedown_vx_u32m1(a, npyv_nlanes_u32 / 2, npyv_nlanes_u32),
        __riscv_vslidedown_vx_u32m1(b, npyv_nlanes_u32 / 2, npyv_nlanes_u32),
        npyv_nlanes_u32 / 2,
        npyv_nlanes_u32
    );
}

NPY_FINLINE npyv_u64 npyv_combineh_u64(npyv_u64 a, npyv_u64 b)
{
    return __riscv_vslideup_vx_u64m1(
        __riscv_vslidedown_vx_u64m1(a, npyv_nlanes_u64 / 2, npyv_nlanes_u64),
        __riscv_vslidedown_vx_u64m1(b, npyv_nlanes_u64 / 2, npyv_nlanes_u64),
        npyv_nlanes_u64 / 2,
        npyv_nlanes_u64
    );
}

NPY_FINLINE npyv_s8 npyv_combineh_s8(npyv_s8 a, npyv_s8 b)
{
    return __riscv_vslideup_vx_i8m1(
        __riscv_vslidedown_vx_i8m1(a, npyv_nlanes_s8 / 2, npyv_nlanes_s8),
        __riscv_vslidedown_vx_i8m1(b, npyv_nlanes_s8 / 2, npyv_nlanes_s8),
        npyv_nlanes_s8 / 2,
        npyv_nlanes_s8
    );
}

NPY_FINLINE npyv_s16 npyv_combineh_s16(npyv_s16 a, npyv_s16 b)
{
    return __riscv_vslideup_vx_i16m1(
        __riscv_vslidedown_vx_i16m1(a, npyv_nlanes_s16 / 2, npyv_nlanes_s16),
        __riscv_vslidedown_vx_i16m1(b, npyv_nlanes_s16 / 2, npyv_nlanes_s16),
        npyv_nlanes_s16 / 2,
        npyv_nlanes_s16
    );
}

NPY_FINLINE npyv_s32 npyv_combineh_s32(npyv_s32 a, npyv_s32 b)
{
    return __riscv_vslideup_vx_i32m1(
        __riscv_vslidedown_vx_i32m1(a, npyv_nlanes_s32 / 2, npyv_nlanes_s32),
        __riscv_vslidedown_vx_i32m1(b, npyv_nlanes_s32 / 2, npyv_nlanes_s32),
        npyv_nlanes_s32 / 2,
        npyv_nlanes_s32
    );
}

NPY_FINLINE npyv_s64 npyv_combineh_s64(npyv_s64 a, npyv_s64 b)
{
    return __riscv_vslideup_vx_i64m1(
        __riscv_vslidedown_vx_i64m1(a, npyv_nlanes_s64 / 2, npyv_nlanes_s64),
        __riscv_vslidedown_vx_i64m1(b, npyv_nlanes_s64 / 2, npyv_nlanes_s64),
        npyv_nlanes_s64 / 2,
        npyv_nlanes_s64
    );
}

NPY_FINLINE npyv_f32 npyv_combineh_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vslideup_vx_f32m1(
        __riscv_vslidedown_vx_f32m1(a, npyv_nlanes_f32 / 2, npyv_nlanes_f32),
        __riscv_vslidedown_vx_f32m1(b, npyv_nlanes_f32 / 2, npyv_nlanes_f32),
        npyv_nlanes_f32 / 2,
        npyv_nlanes_f32
    );
}

NPY_FINLINE npyv_f64 npyv_combineh_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vslideup_vx_f64m1(
        __riscv_vslidedown_vx_f64m1(a, npyv_nlanes_f64 / 2, npyv_nlanes_f64),
        __riscv_vslidedown_vx_f64m1(b, npyv_nlanes_f64 / 2, npyv_nlanes_f64),
        npyv_nlanes_f64 / 2,
        npyv_nlanes_f64
    );
}

// combine two vectors from lower and higher parts of two other vectors
#define NPYV_IMPL_RVV_COMBINE(T_VEC, SFX)                      \
    NPY_FINLINE T_VEC##x2 npyv_combine_##SFX(T_VEC a, T_VEC b) \
    {                                                          \
        return (T_VEC##x2){{                                   \
            npyv_combinel_##SFX(a, b),                         \
            npyv_combineh_##SFX(a, b)                          \
        }};                                                    \
    }

NPYV_IMPL_RVV_COMBINE(npyv_u8,  u8)
NPYV_IMPL_RVV_COMBINE(npyv_s8,  s8)
NPYV_IMPL_RVV_COMBINE(npyv_u16, u16)
NPYV_IMPL_RVV_COMBINE(npyv_s16, s16)
NPYV_IMPL_RVV_COMBINE(npyv_u32, u32)
NPYV_IMPL_RVV_COMBINE(npyv_s32, s32)
NPYV_IMPL_RVV_COMBINE(npyv_u64, u64)
NPYV_IMPL_RVV_COMBINE(npyv_s64, s64)
NPYV_IMPL_RVV_COMBINE(npyv_f32, f32)
NPYV_IMPL_RVV_COMBINE(npyv_f64, f64)
#undef NPYV_IMPL_RVV_COMBINE

// interleave & deinterleave two vectors
#define NPYV_IMPL_RVV_ZIP(T_VEC, SFX, R_SFX, EEW)            \
    NPY_FINLINE T_VEC##x2 npyv_zip_##SFX(T_VEC a, T_VEC b)   \
    {                                                        \
        const int vl = npyv_nlanes_##SFX;                    \
        npyv_lanetype_##SFX v[vl * 2];                       \
        __riscv_vsseg2##EEW(                                 \
            v, __riscv_vcreate_v_##R_SFX##m1x2(a, b), vl     \
        );                                                   \
        return (T_VEC##x2){{                                 \
            __riscv_vl##EEW##_v_##R_SFX##m1(v     , vl),     \
            __riscv_vl##EEW##_v_##R_SFX##m1(v + vl, vl)      \
        }};                                                  \
    }                                                        \
    NPY_FINLINE T_VEC##x2 npyv_unzip_##SFX(T_VEC a, T_VEC b) \
    {                                                        \
        const int vl = npyv_nlanes_##SFX;                    \
        npyv_lanetype_##SFX v[vl * 2];                       \
        __riscv_vs##EEW(v     , a, vl);                      \
        __riscv_vs##EEW(v + vl, b, vl);                      \
        npyv__##SFX##x2 d =                                  \
            __riscv_vlseg2##EEW##_v_##R_SFX##m1x2(v, vl);    \
        return (T_VEC##x2){{                                 \
            __riscv_vget_v_##R_SFX##m1x2_##R_SFX##m1(d, 0),  \
            __riscv_vget_v_##R_SFX##m1x2_##R_SFX##m1(d, 1)   \
        }};                                                  \
    }

NPYV_IMPL_RVV_ZIP(npyv_u8,  u8, u8, e8)
NPYV_IMPL_RVV_ZIP(npyv_s8,  s8, i8, e8)
NPYV_IMPL_RVV_ZIP(npyv_u16, u16, u16, e16)
NPYV_IMPL_RVV_ZIP(npyv_s16, s16, i16, e16)
NPYV_IMPL_RVV_ZIP(npyv_u32, u32, u32, e32)
NPYV_IMPL_RVV_ZIP(npyv_s32, s32, i32, e32)
NPYV_IMPL_RVV_ZIP(npyv_u64, u64, u64, e64)
NPYV_IMPL_RVV_ZIP(npyv_s64, s64, i64, e64)
NPYV_IMPL_RVV_ZIP(npyv_f32, f32, f32, e32)
NPYV_IMPL_RVV_ZIP(npyv_f64, f64, f64, e64)
#undef NPYV_IMPL_RVV_ZIP

// Reverse elements of each 64-bit lane
NPY_FINLINE npyv_u8 npyv_rev64_u8(npyv_u8 a)
{
    vuint8m1_t vid = __riscv_vid_v_u8m1(npyv_nlanes_u8);
    vuint8m1_t sub = __riscv_vadd(__riscv_vsll(__riscv_vsrl(vid, 3, npyv_nlanes_u8), 4, npyv_nlanes_u8), 7, npyv_nlanes_u8);
    vuint8m1_t idxs = __riscv_vsub(sub, vid, npyv_nlanes_u8);
    return __riscv_vrgather(a, idxs, npyv_nlanes_u8);
}
NPY_FINLINE npyv_s8 npyv_rev64_s8(npyv_s8 a)
{ return __riscv_vreinterpret_v_u8m1_i8m1(npyv_rev64_u8(__riscv_vreinterpret_v_i8m1_u8m1(a))); }

NPY_FINLINE npyv_u16 npyv_rev64_u16(npyv_u16 a)
{
    vuint16m1_t vid = __riscv_vid_v_u16m1(npyv_nlanes_u16);
    vuint16m1_t sub = __riscv_vadd(__riscv_vsll(__riscv_vsrl(vid, 2, npyv_nlanes_u16), 3, npyv_nlanes_u16), 3, npyv_nlanes_u16);
    vuint16m1_t idxs = __riscv_vsub(sub, vid, npyv_nlanes_u16);
    return __riscv_vrgather(a, idxs, npyv_nlanes_u16);
}
NPY_FINLINE npyv_s16 npyv_rev64_s16(npyv_s16 a)
{ return __riscv_vreinterpret_v_u16m1_i16m1(npyv_rev64_u16(__riscv_vreinterpret_v_i16m1_u16m1(a))); }

NPY_FINLINE npyv_u32 npyv_rev64_u32(npyv_u32 a)
{
    vuint16mf2_t vid = __riscv_vid_v_u16mf2(npyv_nlanes_u16 / 2);
    vuint16mf2_t sub = __riscv_vadd(__riscv_vsll(__riscv_vsrl(vid, 1, npyv_nlanes_u16 / 2), 2, npyv_nlanes_u16 / 2), 1, npyv_nlanes_u16 / 2);
    vuint16mf2_t idxs = __riscv_vsub(sub, vid, npyv_nlanes_u16 / 2);
    return __riscv_vrgatherei16(a, idxs, npyv_nlanes_u32);
}
NPY_FINLINE npyv_s32 npyv_rev64_s32(npyv_s32 a)
{ return __riscv_vreinterpret_v_u32m1_i32m1(npyv_rev64_u32(__riscv_vreinterpret_v_i32m1_u32m1(a))); }
NPY_FINLINE npyv_f32 npyv_rev64_f32(npyv_f32 a)
{ return __riscv_vreinterpret_v_u32m1_f32m1(npyv_rev64_u32(__riscv_vreinterpret_v_f32m1_u32m1(a))); }

// Permuting the elements of each 128-bit lane by immediate index for
// each element.
#define npyv_permi128_u32(A, E0, E1, E2, E3)               \
    ({                                                     \
        const uint16_t v[] = {                             \
            E0     , E1     , E2     , E3     ,            \
            E0 +  4, E1 +  4, E2 +  4, E3 +  4,            \
            E0 +  8, E1 +  8, E2 +  8, E3 +  8,            \
            E0 + 12, E1 + 12, E2 + 12, E3 + 12,            \
            E0 + 16, E1 + 16, E2 + 16, E3 + 16,            \
            E0 + 20, E1 + 20, E2 + 20, E3 + 20,            \
            E0 + 24, E1 + 24, E2 + 24, E3 + 24,            \
            E0 + 28, E1 + 28, E2 + 28, E3 + 28             \
        };                                                 \
        __riscv_vrgatherei16(                              \
            A, __riscv_vle16_v_u16mf2(v, npyv_nlanes_u32), \
            npyv_nlanes_u32                                \
        );                                                 \
    })
#define npyv_permi128_s32(A, E0, E1, E2, E3) __riscv_vreinterpret_v_u32m1_i32m1(npyv_permi128_u32(__riscv_vreinterpret_v_i32m1_u32m1(A), E0, E1, E2, E3))
#define npyv_permi128_f32(A, E0, E1, E2, E3) __riscv_vreinterpret_v_u32m1_f32m1(npyv_permi128_u32(__riscv_vreinterpret_v_f32m1_u32m1(A), E0, E1, E2, E3))

#define npyv_permi128_u64(A, E0, E1)                       \
    ({                                                     \
        const uint16_t v[] = {                             \
            E0     , E1     ,                              \
            E0 +  2, E1 +  2,                              \
            E0 +  4, E1 +  4,                              \
            E0 +  6, E1 +  6,                              \
            E0 +  8, E1 +  8,                              \
            E0 + 10, E1 + 10,                              \
            E0 + 12, E1 + 12,                              \
            E0 + 14, E1 + 14                               \
        };                                                 \
        __riscv_vrgatherei16(                              \
            A, __riscv_vle16_v_u16mf4(v, npyv_nlanes_u64), \
            npyv_nlanes_u64                                \
        );                                                 \
    })
#define npyv_permi128_s64(A, E0, E1) __riscv_vreinterpret_v_u64m1_i64m1(npyv_permi128_u64(__riscv_vreinterpret_v_i64m1_u64m1(A), E0, E1))
#define npyv_permi128_f64(A, E0, E1) __riscv_vreinterpret_v_u64m1_f64m1(npyv_permi128_u64(__riscv_vreinterpret_v_f64m1_u64m1(A), E0, E1))

#endif // _NPY_SIMD_RVV_REORDER_H
