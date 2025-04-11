#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_CVT_H
#define _NPY_SIMD_RVV_CVT_H

#define npyv_cvt_u8_b8(A)   A
#define npyv_cvt_u16_b16(A) A
#define npyv_cvt_u32_b32(A) A
#define npyv_cvt_u64_b64(A) A
#define npyv_cvt_s8_b8(A)   __riscv_vreinterpret_v_u8m1_i8m1(npyv_cvt_u8_b8(A))
#define npyv_cvt_s16_b16(A) __riscv_vreinterpret_v_u16m1_i16m1(npyv_cvt_u16_b16(A))
#define npyv_cvt_s32_b32(A) __riscv_vreinterpret_v_u32m1_i32m1(npyv_cvt_u32_b32(A))
#define npyv_cvt_s64_b64(A) __riscv_vreinterpret_v_u64m1_i64m1(npyv_cvt_u64_b64(A))
#define npyv_cvt_f32_b32(A) __riscv_vreinterpret_v_u32m1_f32m1(npyv_cvt_u32_b32(A))
#define npyv_cvt_f64_b64(A) __riscv_vreinterpret_v_u64m1_f64m1(npyv_cvt_u64_b64(A))

#define npyv_cvt_b8_u8(A)   A
#define npyv_cvt_b16_u16(A) A
#define npyv_cvt_b32_u32(A) A
#define npyv_cvt_b64_u64(A) A
#define npyv_cvt_b8_s8(A)   npyv_cvt_b8_u8(__riscv_vreinterpret_v_i8m1_u8m1(A))
#define npyv_cvt_b16_s16(A) npyv_cvt_b16_u16(__riscv_vreinterpret_v_i16m1_u16m1(A))
#define npyv_cvt_b32_s32(A) npyv_cvt_b32_u32(__riscv_vreinterpret_v_i32m1_u32m1(A))
#define npyv_cvt_b64_s64(A) npyv_cvt_b64_u64(__riscv_vreinterpret_v_i64m1_u64m1(A))
#define npyv_cvt_b32_f32(A) npyv_cvt_b32_u32(__riscv_vreinterpret_v_f32m1_u32m1(A))
#define npyv_cvt_b64_f64(A) npyv_cvt_b64_u64(__riscv_vreinterpret_v_f64m1_u64m1(A))

#define npyv__from_b8(A)  __riscv_vmseq_vx_u8m1_b8(A, UINT8_MAX, npyv_nlanes_u8)
#define npyv__from_b16(A) __riscv_vmseq_vx_u16m1_b16(A, UINT16_MAX, npyv_nlanes_u16)
#define npyv__from_b32(A) __riscv_vmseq_vx_u32m1_b32(A, UINT32_MAX, npyv_nlanes_u32)
#define npyv__from_b64(A) __riscv_vmseq_vx_u64m1_b64(A, UINT64_MAX, npyv_nlanes_u64)
#define npyv__to_b8(A)  __riscv_vmerge_vxm_u8m1(__riscv_vmv_v_x_u8m1(0, npyv_nlanes_u8), UINT8_MAX, A, npyv_nlanes_u8)
#define npyv__to_b16(A) __riscv_vmerge_vxm_u16m1(__riscv_vmv_v_x_u16m1(0, npyv_nlanes_u16), UINT16_MAX, A, npyv_nlanes_u16)
#define npyv__to_b32(A) __riscv_vmerge_vxm_u32m1(__riscv_vmv_v_x_u32m1(0, npyv_nlanes_u32), UINT32_MAX, A, npyv_nlanes_u32)
#define npyv__to_b64(A) __riscv_vmerge_vxm_u64m1(__riscv_vmv_v_x_u64m1(0, npyv_nlanes_u64), UINT64_MAX, A, npyv_nlanes_u64)

NPY_FINLINE npy_uint64 npyv_tobits_b8(npyv_b8 a)
{ return __riscv_vmv_x(__riscv_vreinterpret_v_b8_u64m1(npyv__from_b8(a))) & (npyv_nlanes_u8 == 64 ? ~0 : (1ULL << npyv_nlanes_u8) - 1); }
NPY_FINLINE npy_uint64 npyv_tobits_b16(npyv_b16 a)
{ return __riscv_vmv_x(__riscv_vreinterpret_v_b16_u64m1(npyv__from_b16(a))) & ((1ULL << npyv_nlanes_u16) - 1); }
NPY_FINLINE npy_uint64 npyv_tobits_b32(npyv_b32 a)
{ return __riscv_vmv_x(__riscv_vreinterpret_v_b32_u64m1(npyv__from_b32(a))) & ((1ULL << npyv_nlanes_u32) - 1); }
NPY_FINLINE npy_uint64 npyv_tobits_b64(npyv_b64 a)
{ return __riscv_vmv_x(__riscv_vreinterpret_v_b64_u64m1(npyv__from_b64(a))) & ((1ULL << npyv_nlanes_u64) - 1); }

//expand
NPY_FINLINE npyv_u16x2 npyv_expand_u16_u8(npyv_u8 data)
{
    vuint16m2_t ext = __riscv_vzext_vf2(data, npyv_nlanes_u8);
    return (npyv_u16x2){{
        __riscv_vget_v_u16m2_u16m1(ext, 0),
        __riscv_vget_v_u16m2_u16m1(ext, 1)
    }};
}

NPY_FINLINE npyv_u32x2 npyv_expand_u32_u16(npyv_u16 data)
{
    vuint32m2_t ext = __riscv_vzext_vf2(data, npyv_nlanes_u16);
    return (npyv_u32x2){{
        __riscv_vget_v_u32m2_u32m1(ext, 0),
        __riscv_vget_v_u32m2_u32m1(ext, 1)
    }};
}

// pack two 16-bit boolean into one 8-bit boolean vector
NPY_FINLINE npyv_b8 npyv_pack_b8_b16(npyv_b16 a, npyv_b16 b)
{
    return npyv__to_b8(__riscv_vreinterpret_v_u64m1_b8(__riscv_vmv_s_x_u64m1(
        npyv_tobits_b16(b) << npyv_nlanes_u16 |
        npyv_tobits_b16(a), 1
    )));
}

// pack four 32-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b32(npyv_b32 a, npyv_b32 b, npyv_b32 c, npyv_b32 d)
{
    return npyv__to_b8(__riscv_vreinterpret_v_u64m1_b8(__riscv_vmv_s_x_u64m1(
        npyv_tobits_b32(d) << (npyv_nlanes_u32 * 3) |
        npyv_tobits_b32(c) << (npyv_nlanes_u32 * 2) |
        npyv_tobits_b32(b) << npyv_nlanes_u32 |
        npyv_tobits_b32(a), 1
    )));
}

 // pack eight 64-bit boolean vectors into one 8-bit boolean vector
NPY_FINLINE npyv_b8
npyv_pack_b8_b64(npyv_b64 a, npyv_b64 b, npyv_b64 c, npyv_b64 d,
                 npyv_b64 e, npyv_b64 f, npyv_b64 g, npyv_b64 h)
{
    return npyv__to_b8(__riscv_vreinterpret_v_u64m1_b8(__riscv_vmv_s_x_u64m1(
        npyv_tobits_b64(h) << (npyv_nlanes_u64 * 7) |
        npyv_tobits_b64(g) << (npyv_nlanes_u64 * 6) |
        npyv_tobits_b64(f) << (npyv_nlanes_u64 * 5) |
        npyv_tobits_b64(e) << (npyv_nlanes_u64 * 4) |
        npyv_tobits_b64(d) << (npyv_nlanes_u64 * 3) |
        npyv_tobits_b64(c) << (npyv_nlanes_u64 * 2) |
        npyv_tobits_b64(b) << npyv_nlanes_u64 |
        npyv_tobits_b64(a), 1
    )));
}

// round to nearest integer
NPY_FINLINE npyv_s32 npyv_round_s32_f32(npyv_f32 a)
{
    // (round-to-nearest-even)
    return __riscv_vfcvt_x_f_v_i32m1(a, npyv_nlanes_s32);
}

NPY_FINLINE npyv_s32 npyv_round_s32_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vfncvt_x_f_w_i32m1(__riscv_vcreate_v_f64m1_f64m2(a, b), npyv_nlanes_s32);
}

#endif // _NPY_SIMD_RVV_CVT_H
