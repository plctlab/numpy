#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_OPERATORS_H
#define _NPY_SIMD_RVV_OPERATORS_H

/***************************
 * Shifting
 ***************************/
// left
NPY_FINLINE npyv_u16 npyv_shl_u16(npyv_u16 a, int16_t c)
{ return __riscv_vsll_vx_u16m1(a, c, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_shl_s16(npyv_s16 a, int16_t c)
{ return __riscv_vsll_vx_i16m1(a, c, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_shl_u32(npyv_u32 a, int32_t c)
{ return __riscv_vsll_vx_u32m1(a, c, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_shl_s32(npyv_s32 a, int32_t c)
{ return __riscv_vsll_vx_i32m1(a, c, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_shl_u64(npyv_u64 a, int64_t c)
{ return __riscv_vsll_vx_u64m1(a, c, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_shl_s64(npyv_s64 a, int64_t c)
{ return __riscv_vsll_vx_i64m1(a, c, npyv_nlanes_s64); }

// left by an immediate constant
NPY_FINLINE npyv_u16 npyv_shli_u16(npyv_u16 a, const int b)
{ return __riscv_vsll_vx_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_shli_s16(npyv_s16 a, const int b)
{ return __riscv_vsll_vx_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_shli_u32(npyv_u32 a, const int b)
{ return __riscv_vsll_vx_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_shli_s32(npyv_s32 a, const int b)
{ return __riscv_vsll_vx_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_shli_u64(npyv_u64 a, const int b)
{ return __riscv_vsll_vx_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_shli_s64(npyv_s64 a, const int b)
{ return __riscv_vsll_vx_i64m1(a, b, npyv_nlanes_s64); }

// right
NPY_FINLINE npyv_u16 npyv_shr_u16(npyv_u16 a, int16_t c)
{ return __riscv_vsrl_vx_u16m1(a, c, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_shr_s16(npyv_s16 a, int16_t c)
{ return __riscv_vsra_vx_i16m1(a, c, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_shr_u32(npyv_u32 a, int32_t c)
{ return __riscv_vsrl_vx_u32m1(a, c, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_shr_s32(npyv_s32 a, int32_t c)
{ return __riscv_vsra_vx_i32m1(a, c, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_shr_u64(npyv_u64 a, int64_t c)
{ return __riscv_vsrl_vx_u64m1(a, c, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_shr_s64(npyv_s64 a, int64_t c)
{ return __riscv_vsra_vx_i64m1(a, c, npyv_nlanes_s64); }

// right by an immediate constant
NPY_FINLINE npyv_u16 npyv_shri_u16(npyv_u16 a, const int b)
{ return __riscv_vsrl_vx_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_shri_s16(npyv_s16 a, const int b)
{ return __riscv_vsra_vx_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_shri_u32(npyv_u32 a, const int b)
{ return __riscv_vsrl_vx_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_shri_s32(npyv_s32 a, const int b)
{ return __riscv_vsra_vx_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_shri_u64(npyv_u64 a, const int b)
{ return __riscv_vsrl_vx_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_shri_s64(npyv_s64 a, const int b)
{ return __riscv_vsra_vx_i64m1(a, b, npyv_nlanes_s64); }

/***************************
 * Logical
 ***************************/
// AND
NPY_FINLINE npyv_u8 npyv_and_u8(npyv_u8 a, npyv_u8 b)
{ return __riscv_vand_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_and_s8(npyv_s8 a, npyv_s8 b)
{ return __riscv_vand_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_and_u16(npyv_u16 a, npyv_u16 b)
{ return __riscv_vand_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_and_s16(npyv_s16 a, npyv_s16 b)
{ return __riscv_vand_vv_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_and_u32(npyv_u32 a, npyv_u32 b)
{ return __riscv_vand_vv_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_and_s32(npyv_s32 a, npyv_s32 b)
{ return __riscv_vand_vv_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_and_u64(npyv_u64 a, npyv_u64 b)
{ return __riscv_vand_vv_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_and_s64(npyv_s64 a, npyv_s64 b)
{ return __riscv_vand_vv_i64m1(a, b, npyv_nlanes_s64); }

NPY_FINLINE npyv_f32 npyv_and_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vand_vv_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(a),
            __riscv_vreinterpret_v_f32m1_u32m1(b),
            npyv_nlanes_f32
        )
    );
}
NPY_FINLINE npyv_f64 npyv_and_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vand_vv_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(a),
            __riscv_vreinterpret_v_f64m1_u64m1(b),
            npyv_nlanes_f64
        )
    );
}

#define npyv_and_b8 npyv_and_u8
#define npyv_and_b16 npyv_and_u16
#define npyv_and_b32 npyv_and_u32
#define npyv_and_b64 npyv_and_u64

// OR
NPY_FINLINE npyv_u8 npyv_or_u8(npyv_u8 a, npyv_u8 b)
{ return __riscv_vor_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_or_s8(npyv_s8 a, npyv_s8 b)
{ return __riscv_vor_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_or_u16(npyv_u16 a, npyv_u16 b)
{ return __riscv_vor_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_or_s16(npyv_s16 a, npyv_s16 b)
{ return __riscv_vor_vv_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_or_u32(npyv_u32 a, npyv_u32 b)
{ return __riscv_vor_vv_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_or_s32(npyv_s32 a, npyv_s32 b)
{ return __riscv_vor_vv_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_or_u64(npyv_u64 a, npyv_u64 b)
{ return __riscv_vor_vv_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_or_s64(npyv_s64 a, npyv_s64 b)
{ return __riscv_vor_vv_i64m1(a, b, npyv_nlanes_s64); }

NPY_FINLINE npyv_f32 npyv_or_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vor_vv_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(a),
            __riscv_vreinterpret_v_f32m1_u32m1(b),
            npyv_nlanes_f32
        )
    );
}
NPY_FINLINE npyv_f64 npyv_or_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vor_vv_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(a),
            __riscv_vreinterpret_v_f64m1_u64m1(b),
            npyv_nlanes_f64
        )
    );
}

#define npyv_or_b8 npyv_or_u8
#define npyv_or_b16 npyv_or_u16
#define npyv_or_b32 npyv_or_u32
#define npyv_or_b64 npyv_or_u64

// XOR
NPY_FINLINE npyv_u8 npyv_xor_u8(npyv_u8 a, npyv_u8 b)
{ return __riscv_vxor_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_xor_s8(npyv_s8 a, npyv_s8 b)
{ return __riscv_vxor_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_xor_u16(npyv_u16 a, npyv_u16 b)
{ return __riscv_vxor_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_xor_s16(npyv_s16 a, npyv_s16 b)
{ return __riscv_vxor_vv_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_xor_u32(npyv_u32 a, npyv_u32 b)
{ return __riscv_vxor_vv_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_xor_s32(npyv_s32 a, npyv_s32 b)
{ return __riscv_vxor_vv_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_xor_u64(npyv_u64 a, npyv_u64 b)
{ return __riscv_vxor_vv_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_xor_s64(npyv_s64 a, npyv_s64 b)
{ return __riscv_vxor_vv_i64m1(a, b, npyv_nlanes_s64); }

NPY_FINLINE npyv_f32 npyv_xor_f32(npyv_f32 a, npyv_f32 b)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vxor_vv_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(a),
            __riscv_vreinterpret_v_f32m1_u32m1(b),
            npyv_nlanes_f32
        )
    );
}
NPY_FINLINE npyv_f64 npyv_xor_f64(npyv_f64 a, npyv_f64 b)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vxor_vv_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(a),
            __riscv_vreinterpret_v_f64m1_u64m1(b),
            npyv_nlanes_f64
        )
    );
}

#define npyv_xor_b8 npyv_xor_u8
#define npyv_xor_b16 npyv_xor_u16
#define npyv_xor_b32 npyv_xor_u32
#define npyv_xor_b64 npyv_xor_u64

// NOT
NPY_FINLINE npyv_u8 npyv_not_u8(npyv_u8 a)
{ return __riscv_vnot_v_u8m1(a, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_not_s8(npyv_s8 a)
{ return __riscv_vnot_v_i8m1(a, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_not_u16(npyv_u16 a)
{ return __riscv_vnot_v_u16m1(a, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_not_s16(npyv_s16 a)
{ return __riscv_vnot_v_i16m1(a, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_not_u32(npyv_u32 a)
{ return __riscv_vnot_v_u32m1(a, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_not_s32(npyv_s32 a)
{ return __riscv_vnot_v_i32m1(a, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_not_u64(npyv_u64 a)
{ return __riscv_vnot_v_u64m1(a, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_not_s64(npyv_s64 a)
{ return __riscv_vnot_v_i64m1(a, npyv_nlanes_s64); }

NPY_FINLINE npyv_f32 npyv_not_f32(npyv_f32 a)
{
    return __riscv_vreinterpret_v_u32m1_f32m1(
        __riscv_vnot_v_u32m1(
            __riscv_vreinterpret_v_f32m1_u32m1(a),
            npyv_nlanes_f32
        )
    );
}
NPY_FINLINE npyv_f64 npyv_not_f64(npyv_f64 a)
{
    return __riscv_vreinterpret_v_u64m1_f64m1(
        __riscv_vnot_v_u64m1(
            __riscv_vreinterpret_v_f64m1_u64m1(a),
            npyv_nlanes_f64
        )
    );
}

#define npyv_not_b8 npyv_not_u8
#define npyv_not_b16 npyv_not_u16
#define npyv_not_b32 npyv_not_u32
#define npyv_not_b64 npyv_not_u64

// ANDC, ORC and XNOR
NPY_FINLINE npyv_u8 npyv_andc_u8(npyv_u8 a, npyv_u8 b)
{ return __riscv_vand_vv_u8m1(a, __riscv_vnot_v_u8m1(b, npyv_nlanes_u8), npyv_nlanes_u8); }

#define npyv_andc_b8 npyv_andc_u8
NPY_FINLINE npyv_b8 npyv_orc_b8(npyv_b8 a, npyv_b8 b)
{ return __riscv_vor_vv_u8m1(a, __riscv_vnot_v_u8m1(b, npyv_nlanes_u8), npyv_nlanes_u8); }
NPY_FINLINE npyv_b8 npyv_xnor_b8(npyv_b8 a, npyv_b8 b)
{ return __riscv_vnot_v_u8m1(__riscv_vxor_vv_u8m1(a, b, npyv_nlanes_u8), npyv_nlanes_u8); }

/***************************
 * Comparison
 ***************************/
// equal
NPY_FINLINE npyv_b8 npyv_cmpeq_u8(npyv_u8 a, npyv_u8 b)
{ return npyv__to_b8(__riscv_vmseq_vv_u8m1_b8(a, b, npyv_nlanes_u8)); }
NPY_FINLINE npyv_b8 npyv_cmpeq_s8(npyv_s8 a, npyv_s8 b)
{ return npyv__to_b8(__riscv_vmseq_vv_i8m1_b8(a, b, npyv_nlanes_s8)); }
NPY_FINLINE npyv_b16 npyv_cmpeq_u16(npyv_u16 a, npyv_u16 b)
{ return npyv__to_b16(__riscv_vmseq_vv_u16m1_b16(a, b, npyv_nlanes_u16)); }
NPY_FINLINE npyv_b16 npyv_cmpeq_s16(npyv_s16 a, npyv_s16 b)
{ return npyv__to_b16(__riscv_vmseq_vv_i16m1_b16(a, b, npyv_nlanes_s16)); }
NPY_FINLINE npyv_b32 npyv_cmpeq_u32(npyv_u32 a, npyv_u32 b)
{ return npyv__to_b32(__riscv_vmseq_vv_u32m1_b32(a, b, npyv_nlanes_u32)); }
NPY_FINLINE npyv_b32 npyv_cmpeq_s32(npyv_s32 a, npyv_s32 b)
{ return npyv__to_b32(__riscv_vmseq_vv_i32m1_b32(a, b, npyv_nlanes_s32)); }
NPY_FINLINE npyv_b64 npyv_cmpeq_u64(npyv_u64 a, npyv_u64 b)
{ return npyv__to_b64(__riscv_vmseq_vv_u64m1_b64(a, b, npyv_nlanes_u64)); }
NPY_FINLINE npyv_b64 npyv_cmpeq_s64(npyv_s64 a, npyv_s64 b)
{ return npyv__to_b64(__riscv_vmseq_vv_i64m1_b64(a, b, npyv_nlanes_s64)); }
NPY_FINLINE npyv_b32 npyv_cmpeq_f32(npyv_f32 a, npyv_f32 b)
{ return npyv__to_b32(__riscv_vmfeq_vv_f32m1_b32(a, b, npyv_nlanes_f32)); }
NPY_FINLINE npyv_b64 npyv_cmpeq_f64(npyv_f64 a, npyv_f64 b)
{ return npyv__to_b64(__riscv_vmfeq_vv_f64m1_b64(a, b, npyv_nlanes_f64)); }

// not Equal
NPY_FINLINE npyv_b8 npyv_cmpneq_u8(npyv_u8 a, npyv_u8 b)
{ return npyv__to_b8(__riscv_vmsne_vv_u8m1_b8(a, b, npyv_nlanes_u8)); }
NPY_FINLINE npyv_b8 npyv_cmpneq_s8(npyv_s8 a, npyv_s8 b)
{ return npyv__to_b8(__riscv_vmsne_vv_i8m1_b8(a, b, npyv_nlanes_s8)); }
NPY_FINLINE npyv_b16 npyv_cmpneq_u16(npyv_u16 a, npyv_u16 b)
{ return npyv__to_b16(__riscv_vmsne_vv_u16m1_b16(a, b, npyv_nlanes_u16)); }
NPY_FINLINE npyv_b16 npyv_cmpneq_s16(npyv_s16 a, npyv_s16 b)
{ return npyv__to_b16(__riscv_vmsne_vv_i16m1_b16(a, b, npyv_nlanes_s16)); }
NPY_FINLINE npyv_b32 npyv_cmpneq_u32(npyv_u32 a, npyv_u32 b)
{ return npyv__to_b32(__riscv_vmsne_vv_u32m1_b32(a, b, npyv_nlanes_u32)); }
NPY_FINLINE npyv_b32 npyv_cmpneq_s32(npyv_s32 a, npyv_s32 b)
{ return npyv__to_b32(__riscv_vmsne_vv_i32m1_b32(a, b, npyv_nlanes_s32)); }
NPY_FINLINE npyv_b64 npyv_cmpneq_u64(npyv_u64 a, npyv_u64 b)
{ return npyv__to_b64(__riscv_vmsne_vv_u64m1_b64(a, b, npyv_nlanes_u64)); }
NPY_FINLINE npyv_b64 npyv_cmpneq_s64(npyv_s64 a, npyv_s64 b)
{ return npyv__to_b64(__riscv_vmsne_vv_i64m1_b64(a, b, npyv_nlanes_s64)); }
NPY_FINLINE npyv_b32 npyv_cmpneq_f32(npyv_f32 a, npyv_f32 b)
{ return npyv__to_b32(__riscv_vmfne_vv_f32m1_b32(a, b, npyv_nlanes_f32)); }
NPY_FINLINE npyv_b64 npyv_cmpneq_f64(npyv_f64 a, npyv_f64 b)
{ return npyv__to_b64(__riscv_vmfne_vv_f64m1_b64(a, b, npyv_nlanes_f64)); }

// greater than
NPY_FINLINE npyv_b8 npyv_cmpgt_u8(npyv_u8 a, npyv_u8 b)
{ return npyv__to_b8(__riscv_vmsgtu_vv_u8m1_b8(a, b, npyv_nlanes_u8)); }
NPY_FINLINE npyv_b8 npyv_cmpgt_s8(npyv_s8 a, npyv_s8 b)
{ return npyv__to_b8(__riscv_vmsgt_vv_i8m1_b8(a, b, npyv_nlanes_s8)); }
NPY_FINLINE npyv_b16 npyv_cmpgt_u16(npyv_u16 a, npyv_u16 b)
{ return npyv__to_b16(__riscv_vmsgtu_vv_u16m1_b16(a, b, npyv_nlanes_u16)); }
NPY_FINLINE npyv_b16 npyv_cmpgt_s16(npyv_s16 a, npyv_s16 b)
{ return npyv__to_b16(__riscv_vmsgt_vv_i16m1_b16(a, b, npyv_nlanes_s16)); }
NPY_FINLINE npyv_b32 npyv_cmpgt_u32(npyv_u32 a, npyv_u32 b)
{ return npyv__to_b32(__riscv_vmsgtu_vv_u32m1_b32(a, b, npyv_nlanes_u32)); }
NPY_FINLINE npyv_b32 npyv_cmpgt_s32(npyv_s32 a, npyv_s32 b)
{ return npyv__to_b32(__riscv_vmsgt_vv_i32m1_b32(a, b, npyv_nlanes_s32)); }
NPY_FINLINE npyv_b64 npyv_cmpgt_u64(npyv_u64 a, npyv_u64 b)
{ return npyv__to_b64(__riscv_vmsgtu_vv_u64m1_b64(a, b, npyv_nlanes_u64)); }
NPY_FINLINE npyv_b64 npyv_cmpgt_s64(npyv_s64 a, npyv_s64 b)
{ return npyv__to_b64(__riscv_vmsgt_vv_i64m1_b64(a, b, npyv_nlanes_s64)); }
NPY_FINLINE npyv_b32 npyv_cmpgt_f32(npyv_f32 a, npyv_f32 b)
{ return npyv__to_b32(__riscv_vmfgt_vv_f32m1_b32(a, b, npyv_nlanes_f32)); }
NPY_FINLINE npyv_b64 npyv_cmpgt_f64(npyv_f64 a, npyv_f64 b)
{ return npyv__to_b64(__riscv_vmfgt_vv_f64m1_b64(a, b, npyv_nlanes_f64)); }

// greater than or equal
NPY_FINLINE npyv_b8 npyv_cmpge_u8(npyv_u8 a, npyv_u8 b)
{ return npyv__to_b8(__riscv_vmsgeu_vv_u8m1_b8(a, b, npyv_nlanes_u8)); }
NPY_FINLINE npyv_b8 npyv_cmpge_s8(npyv_s8 a, npyv_s8 b)
{ return npyv__to_b8(__riscv_vmsge_vv_i8m1_b8(a, b, npyv_nlanes_s8)); }
NPY_FINLINE npyv_b16 npyv_cmpge_u16(npyv_u16 a, npyv_u16 b)
{ return npyv__to_b16(__riscv_vmsgeu_vv_u16m1_b16(a, b, npyv_nlanes_u16)); }
NPY_FINLINE npyv_b16 npyv_cmpge_s16(npyv_s16 a, npyv_s16 b)
{ return npyv__to_b16(__riscv_vmsge_vv_i16m1_b16(a, b, npyv_nlanes_s16)); }
NPY_FINLINE npyv_b32 npyv_cmpge_u32(npyv_u32 a, npyv_u32 b)
{ return npyv__to_b32(__riscv_vmsgeu_vv_u32m1_b32(a, b, npyv_nlanes_u32)); }
NPY_FINLINE npyv_b32 npyv_cmpge_s32(npyv_s32 a, npyv_s32 b)
{ return npyv__to_b32(__riscv_vmsge_vv_i32m1_b32(a, b, npyv_nlanes_s32)); }
NPY_FINLINE npyv_b64 npyv_cmpge_u64(npyv_u64 a, npyv_u64 b)
{ return npyv__to_b64(__riscv_vmsgeu_vv_u64m1_b64(a, b, npyv_nlanes_u64)); }
NPY_FINLINE npyv_b64 npyv_cmpge_s64(npyv_s64 a, npyv_s64 b)
{ return npyv__to_b64(__riscv_vmsge_vv_i64m1_b64(a, b, npyv_nlanes_s64)); }
NPY_FINLINE npyv_b32 npyv_cmpge_f32(npyv_f32 a, npyv_f32 b)
{ return npyv__to_b32(__riscv_vmfge_vv_f32m1_b32(a, b, npyv_nlanes_f32)); }
NPY_FINLINE npyv_b64 npyv_cmpge_f64(npyv_f64 a, npyv_f64 b)
{ return npyv__to_b64(__riscv_vmfge_vv_f64m1_b64(a, b, npyv_nlanes_f64)); }

// less than
#define npyv_cmplt_u8(A, B)  npyv_cmpgt_u8(B, A)
#define npyv_cmplt_s8(A, B)  npyv_cmpgt_s8(B, A)
#define npyv_cmplt_u16(A, B) npyv_cmpgt_u16(B, A)
#define npyv_cmplt_s16(A, B) npyv_cmpgt_s16(B, A)
#define npyv_cmplt_u32(A, B) npyv_cmpgt_u32(B, A)
#define npyv_cmplt_s32(A, B) npyv_cmpgt_s32(B, A)
#define npyv_cmplt_u64(A, B) npyv_cmpgt_u64(B, A)
#define npyv_cmplt_s64(A, B) npyv_cmpgt_s64(B, A)
#define npyv_cmplt_f32(A, B) npyv_cmpgt_f32(B, A)
#define npyv_cmplt_f64(A, B) npyv_cmpgt_f64(B, A)

// less than or equal
#define npyv_cmple_u8(A, B)  npyv_cmpge_u8(B, A)
#define npyv_cmple_s8(A, B)  npyv_cmpge_s8(B, A)
#define npyv_cmple_u16(A, B) npyv_cmpge_u16(B, A)
#define npyv_cmple_s16(A, B) npyv_cmpge_s16(B, A)
#define npyv_cmple_u32(A, B) npyv_cmpge_u32(B, A)
#define npyv_cmple_s32(A, B) npyv_cmpge_s32(B, A)
#define npyv_cmple_u64(A, B) npyv_cmpge_u64(B, A)
#define npyv_cmple_s64(A, B) npyv_cmpge_s64(B, A)
#define npyv_cmple_f32(A, B) npyv_cmpge_f32(B, A)
#define npyv_cmple_f64(A, B) npyv_cmpge_f64(B, A)

// check special cases
NPY_FINLINE npyv_b32 npyv_notnan_f32(npyv_f32 a)
{ return npyv__to_b32(__riscv_vmfeq_vv_f32m1_b32(a, a, npyv_nlanes_f32)); }
NPY_FINLINE npyv_b64 npyv_notnan_f64(npyv_f64 a)
{ return npyv__to_b64(__riscv_vmfeq_vv_f64m1_b64(a, a, npyv_nlanes_f64)); }

// Test cross all vector lanes
// any: returns true if any of the elements is not equal to zero
// all: returns true if all elements are not equal to zero
NPY_FINLINE bool npyv_any_u8(npyv_u8 a)
{ return __riscv_vfirst(__riscv_vmsne(a, 0, npyv_nlanes_u8), npyv_nlanes_u8) != -1; }
NPY_FINLINE bool npyv_all_u8(npyv_u8 a)
{ return __riscv_vfirst(__riscv_vmseq(a, 0, npyv_nlanes_u8), npyv_nlanes_u8) == -1; }
NPY_FINLINE bool npyv_any_u16(npyv_u16 a)
{ return __riscv_vfirst(__riscv_vmsne(a, 0, npyv_nlanes_u16), npyv_nlanes_u16) != -1; }
NPY_FINLINE bool npyv_all_u16(npyv_u16 a)
{ return __riscv_vfirst(__riscv_vmseq(a, 0, npyv_nlanes_u16), npyv_nlanes_u16) == -1; }
NPY_FINLINE bool npyv_any_u32(npyv_u32 a)
{ return __riscv_vfirst(__riscv_vmsne(a, 0, npyv_nlanes_u32), npyv_nlanes_u32) != -1; }
NPY_FINLINE bool npyv_all_u32(npyv_u32 a)
{ return __riscv_vfirst(__riscv_vmseq(a, 0, npyv_nlanes_u32), npyv_nlanes_u32) == -1; }
NPY_FINLINE bool npyv_any_u64(npyv_u64 a)
{ return __riscv_vfirst(__riscv_vmsne(a, 0, npyv_nlanes_u64), npyv_nlanes_u64) != -1; }
NPY_FINLINE bool npyv_all_u64(npyv_u64 a)
{ return __riscv_vfirst(__riscv_vmseq(a, 0, npyv_nlanes_u64), npyv_nlanes_u64) == -1; }

#define npyv_any_b8 npyv_any_u8
#define npyv_all_b8 npyv_all_u8
#define npyv_any_b16 npyv_any_u16
#define npyv_all_b16 npyv_all_u16
#define npyv_any_b32 npyv_any_u32
#define npyv_all_b32 npyv_all_u32
#define npyv_any_b64 npyv_any_u64
#define npyv_all_b64 npyv_all_u64

NPY_FINLINE bool npyv_any_s8(npyv_s8 a)
{ return npyv_any_u8(npyv_reinterpret_u8_s8(a)); }
NPY_FINLINE bool npyv_all_s8(npyv_s8 a)
{ return npyv_all_u8(npyv_reinterpret_u8_s8(a)); }
NPY_FINLINE bool npyv_any_s16(npyv_s16 a)
{ return npyv_any_u16(npyv_reinterpret_u16_s16(a)); }
NPY_FINLINE bool npyv_all_s16(npyv_s16 a)
{ return npyv_all_u16(npyv_reinterpret_u16_s16(a)); }
NPY_FINLINE bool npyv_any_s32(npyv_s32 a)
{ return npyv_any_u32(npyv_reinterpret_u32_s32(a)); }
NPY_FINLINE bool npyv_all_s32(npyv_s32 a)
{ return npyv_all_u32(npyv_reinterpret_u32_s32(a)); }
NPY_FINLINE bool npyv_any_s64(npyv_s64 a)
{ return npyv_any_u64(npyv_reinterpret_u64_s64(a)); }
NPY_FINLINE bool npyv_all_s64(npyv_s64 a)
{ return npyv_all_u64(npyv_reinterpret_u64_s64(a)); }

NPY_FINLINE bool npyv_any_f32(npyv_f32 a)
{ return npyv_any_u32(npyv_reinterpret_u32_f32(__riscv_vfabs(a, npyv_nlanes_f32))); }
NPY_FINLINE bool npyv_all_f32(npyv_f32 a)
{ return npyv_all_u32(npyv_reinterpret_u32_f32(__riscv_vfabs(a, npyv_nlanes_f32))); }
NPY_FINLINE bool npyv_any_f64(npyv_f64 a)
{ return npyv_any_u64(npyv_reinterpret_u64_f64(__riscv_vfabs(a, npyv_nlanes_f64))); }
NPY_FINLINE bool npyv_all_f64(npyv_f64 a)
{ return npyv_all_u64(npyv_reinterpret_u64_f64(__riscv_vfabs(a, npyv_nlanes_f64))); }

#endif // _NPY_SIMD_RVV_OPERATORS_H
