#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_ARITHMETIC_H
#define _NPY_SIMD_RVV_ARITHMETIC_H

/***************************
 * Addition
 ***************************/
// non-saturated
NPY_FINLINE npyv_u8 npyv_add_u8(npyv_u8 a, npyv_u8 b) { return __riscv_vadd_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_add_s8(npyv_s8 a, npyv_s8 b) { return __riscv_vadd_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_add_u16(npyv_u16 a, npyv_u16 b) { return __riscv_vadd_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_add_s16(npyv_s16 a, npyv_s16 b) { return __riscv_vadd_vv_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_add_u32(npyv_u32 a, npyv_u32 b) { return __riscv_vadd_vv_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_add_s32(npyv_s32 a, npyv_s32 b) { return __riscv_vadd_vv_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_add_u64(npyv_u64 a, npyv_u64 b) { return __riscv_vadd_vv_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_add_s64(npyv_s64 a, npyv_s64 b) { return __riscv_vadd_vv_i64m1(a, b, npyv_nlanes_s64); }
NPY_FINLINE npyv_f32 npyv_add_f32(npyv_f32 a, npyv_f32 b) { return __riscv_vfadd_vv_f32m1(a, b, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_add_f64(npyv_f64 a, npyv_f64 b) { return __riscv_vfadd_vv_f64m1(a, b, npyv_nlanes_f64); }

// saturated
NPY_FINLINE npyv_u8 npyv_adds_u8(npyv_u8 a, npyv_u8 b) { return __riscv_vsaddu_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_adds_s8(npyv_s8 a, npyv_s8 b) { return __riscv_vsadd_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_adds_u16(npyv_u16 a, npyv_u16 b) { return __riscv_vsaddu_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_adds_s16(npyv_s16 a, npyv_s16 b) { return __riscv_vsadd_vv_i16m1(a, b, npyv_nlanes_s16); }

/***************************
 * Subtraction
 ***************************/
// non-saturated
NPY_FINLINE npyv_u8 npyv_sub_u8(npyv_u8 a, npyv_u8 b) { return __riscv_vsub_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_sub_s8(npyv_s8 a, npyv_s8 b) { return __riscv_vsub_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_sub_u16(npyv_u16 a, npyv_u16 b) { return __riscv_vsub_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_sub_s16(npyv_s16 a, npyv_s16 b) { return __riscv_vsub_vv_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_sub_u32(npyv_u32 a, npyv_u32 b) { return __riscv_vsub_vv_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_sub_s32(npyv_s32 a, npyv_s32 b) { return __riscv_vsub_vv_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_u64 npyv_sub_u64(npyv_u64 a, npyv_u64 b) { return __riscv_vsub_vv_u64m1(a, b, npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_sub_s64(npyv_s64 a, npyv_s64 b) { return __riscv_vsub_vv_i64m1(a, b, npyv_nlanes_s64); }
NPY_FINLINE npyv_f32 npyv_sub_f32(npyv_f32 a, npyv_f32 b) { return __riscv_vfsub_vv_f32m1(a, b, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_sub_f64(npyv_f64 a, npyv_f64 b) { return __riscv_vfsub_vv_f64m1(a, b, npyv_nlanes_f64); }

// saturated
NPY_FINLINE npyv_u8 npyv_subs_u8(npyv_u8 a, npyv_u8 b) { return __riscv_vssubu_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_subs_s8(npyv_s8 a, npyv_s8 b) { return __riscv_vssub_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_subs_u16(npyv_u16 a, npyv_u16 b) { return __riscv_vssubu_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_subs_s16(npyv_s16 a, npyv_s16 b) { return __riscv_vssub_vv_i16m1(a, b, npyv_nlanes_s16); }

/***************************
 * Multiplication
 ***************************/
// non-saturated
NPY_FINLINE npyv_u8 npyv_mul_u8(npyv_u8 a, npyv_u8 b) { return __riscv_vmul_vv_u8m1(a, b, npyv_nlanes_u8); }
NPY_FINLINE npyv_s8 npyv_mul_s8(npyv_s8 a, npyv_s8 b) { return __riscv_vmul_vv_i8m1(a, b, npyv_nlanes_s8); }
NPY_FINLINE npyv_u16 npyv_mul_u16(npyv_u16 a, npyv_u16 b) { return __riscv_vmul_vv_u16m1(a, b, npyv_nlanes_u16); }
NPY_FINLINE npyv_s16 npyv_mul_s16(npyv_s16 a, npyv_s16 b) { return __riscv_vmul_vv_i16m1(a, b, npyv_nlanes_s16); }
NPY_FINLINE npyv_u32 npyv_mul_u32(npyv_u32 a, npyv_u32 b) { return __riscv_vmul_vv_u32m1(a, b, npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_mul_s32(npyv_s32 a, npyv_s32 b) { return __riscv_vmul_vv_i32m1(a, b, npyv_nlanes_s32); }
NPY_FINLINE npyv_f32 npyv_mul_f32(npyv_f32 a, npyv_f32 b) { return __riscv_vfmul_vv_f32m1(a, b, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_mul_f64(npyv_f64 a, npyv_f64 b) { return __riscv_vfmul_vv_f64m1(a, b, npyv_nlanes_f64); }

/***************************
 * Integer Division
 ***************************/
// See simd/intdiv.h for more clarification
// divide each unsigned 8-bit element by a precomputed divisor
NPY_FINLINE npyv_u8 npyv_divc_u8(npyv_u8 a, const npyv_u8x3 divisor)
{
    // high part of unsigned multiplication
    vuint8m1_t mulhi = __riscv_vmulhu(a, divisor.val[0], npyv_nlanes_u8);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    vuint8m1_t q     = __riscv_vsub(a, mulhi, npyv_nlanes_u8);
               q     = __riscv_vsrl(q, divisor.val[1], npyv_nlanes_u8);
               q     = __riscv_vadd(mulhi, q, npyv_nlanes_u8);
               q     = __riscv_vsrl(q, divisor.val[2], npyv_nlanes_u8);

    return     q;
}
// divide each signed 8-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s8 npyv_divc_s8(npyv_s8 a, const npyv_s8x3 divisor)
{
    vint8m1_t mulhi = __riscv_vmulh(a, divisor.val[0], npyv_nlanes_s8);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    vint8m1_t q     = __riscv_vsra(__riscv_vadd(a, mulhi, npyv_nlanes_s8), __riscv_vreinterpret_v_i8m1_u8m1(divisor.val[1]), npyv_nlanes_s8);
              q     = __riscv_vsub(q, __riscv_vsra(a, 7, npyv_nlanes_s8), npyv_nlanes_s8);
              q     = __riscv_vsub(__riscv_vxor(q, divisor.val[2], npyv_nlanes_s8), divisor.val[2], npyv_nlanes_s8);
    return    q;
}
// divide each unsigned 16-bit element by a precomputed divisor
NPY_FINLINE npyv_u16 npyv_divc_u16(npyv_u16 a, const npyv_u16x3 divisor)
{
    // high part of unsigned multiplication
    vuint16m1_t mulhi = __riscv_vmulhu(a, divisor.val[0], npyv_nlanes_u16);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    vuint16m1_t q     = __riscv_vsub(a, mulhi, npyv_nlanes_u16);
                q     = __riscv_vsrl(q, divisor.val[1], npyv_nlanes_u16);
                q     = __riscv_vadd(mulhi, q, npyv_nlanes_u16);
                q     = __riscv_vsrl(q, divisor.val[2], npyv_nlanes_u16);
    return      q;
}
// divide each signed 16-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s16 npyv_divc_s16(npyv_s16 a, const npyv_s16x3 divisor)
{
    // high part of signed multiplication
    vint16m1_t mulhi = __riscv_vmulh(a, divisor.val[0], npyv_nlanes_s16);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    vint16m1_t q     = __riscv_vsra(__riscv_vadd(a, mulhi, npyv_nlanes_s16), __riscv_vreinterpret_v_i16m1_u16m1(divisor.val[1]), npyv_nlanes_s16);
               q     = __riscv_vsub(q, __riscv_vsra(a, 15, npyv_nlanes_s16), npyv_nlanes_s16);
               q     = __riscv_vsub(__riscv_vxor(q, divisor.val[2], npyv_nlanes_s16), divisor.val[2], npyv_nlanes_s16);
    return     q;
}
// divide each unsigned 32-bit element by a precomputed divisor
NPY_FINLINE npyv_u32 npyv_divc_u32(npyv_u32 a, const npyv_u32x3 divisor)
{
    // high part of unsigned multiplication
    vuint32m1_t mulhi = __riscv_vmulhu(a, divisor.val[0], npyv_nlanes_u32);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    vuint32m1_t q     = __riscv_vsub(a, mulhi, npyv_nlanes_u32);
                q     = __riscv_vsrl(q, divisor.val[1], npyv_nlanes_u32);
                q     = __riscv_vadd(mulhi, q, npyv_nlanes_u32);
                q     = __riscv_vsrl(q, divisor.val[2], npyv_nlanes_u32);

    return     q;
}
// divide each signed 32-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s32 npyv_divc_s32(npyv_s32 a, const npyv_s32x3 divisor)
{
    // high part of signed multiplication
    vint32m1_t mulhi = __riscv_vmulh(a, divisor.val[0], npyv_nlanes_s32);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    vint32m1_t q     = __riscv_vsra(__riscv_vadd(a, mulhi, npyv_nlanes_s32), __riscv_vreinterpret_v_i32m1_u32m1(divisor.val[1]), npyv_nlanes_s32);
               q     = __riscv_vsub(q, __riscv_vsra(a, 31, npyv_nlanes_s32), npyv_nlanes_s32);
               q     = __riscv_vsub(__riscv_vxor(q, divisor.val[2], npyv_nlanes_s32), divisor.val[2], npyv_nlanes_s32);
    return     q;
}
// divide each unsigned 64-bit element by a precomputed divisor
NPY_FINLINE npyv_u64 npyv_divc_u64(npyv_u64 a, const npyv_u64x3 divisor)
{
    // high part of unsigned multiplication
    vuint64m1_t mulhi = __riscv_vmulhu(a, divisor.val[0], npyv_nlanes_u64);
    // floor(a/d) = (mulhi + ((a-mulhi) >> sh1)) >> sh2
    vuint64m1_t q     = __riscv_vsub(a, mulhi, npyv_nlanes_u64);
                q     = __riscv_vsrl(q, divisor.val[1], npyv_nlanes_u64);
                q     = __riscv_vadd(mulhi, q, npyv_nlanes_u64);
                q     = __riscv_vsrl(q, divisor.val[2], npyv_nlanes_u64);

    return     q;
}
// divide each signed 64-bit element by a precomputed divisor (round towards zero)
NPY_FINLINE npyv_s64 npyv_divc_s64(npyv_s64 a, const npyv_s64x3 divisor)
{
    // high part of signed multiplication
    vint64m1_t mulhi = __riscv_vmulh(a, divisor.val[0], npyv_nlanes_s64);
    // q          = ((a + mulhi) >> sh1) - XSIGN(a)
    // trunc(a/d) = (q ^ dsign) - dsign
    vint64m1_t q     = __riscv_vsra(__riscv_vadd(a, mulhi, npyv_nlanes_s64), __riscv_vreinterpret_v_i64m1_u64m1(divisor.val[1]), npyv_nlanes_s64);
               q     = __riscv_vsub(q, __riscv_vsra(a, 63, npyv_nlanes_s64), npyv_nlanes_s64);
               q     = __riscv_vsub(__riscv_vxor(q, divisor.val[2], npyv_nlanes_s64), divisor.val[2], npyv_nlanes_s64);
    return     q;
}

/***************************
 * Division
 ***************************/
NPY_FINLINE npyv_f32 npyv_div_f32(npyv_f32 a, npyv_f32 b) { return __riscv_vfdiv_vv_f32m1(a, b, npyv_nlanes_f32); }
NPY_FINLINE npyv_f64 npyv_div_f64(npyv_f64 a, npyv_f64 b) { return __riscv_vfdiv_vv_f64m1(a, b, npyv_nlanes_f64); }

/***************************
 * FUSED F32
 ***************************/
// multiply and add, a*b + c
NPY_FINLINE npyv_f32 npyv_muladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return __riscv_vfmadd_vv_f32m1(a, b, c, npyv_nlanes_f32); }
// multiply and subtract, a*b - c
NPY_FINLINE npyv_f32 npyv_mulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return __riscv_vfmsub_vv_f32m1(a, b, c, npyv_nlanes_f32); }
// negate multiply and add, -(a*b) + c
NPY_FINLINE npyv_f32 npyv_nmuladd_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return __riscv_vfnmsub_vv_f32m1(a, b, c, npyv_nlanes_f32); }
// negate multiply and subtract, -(a*b) - c
NPY_FINLINE npyv_f32 npyv_nmulsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return __riscv_vfnmadd_vv_f32m1(a, b, c, npyv_nlanes_f32); }

// multiply, add for odd elements and subtract even elements.
// (a * b) -+ c
NPY_FINLINE npyv_f32 npyv_muladdsub_f32(npyv_f32 a, npyv_f32 b, npyv_f32 c)
{ return npyv_muladd_f32(a, b, __riscv_vfneg_v_f32m1_mu(__riscv_vreinterpret_v_u8m1_b32(__riscv_vmv_v_x_u8m1(0x55, npyv_nlanes_u8)), c, c, npyv_nlanes_f32)); }

/***************************
 * FUSED F64
 ***************************/
NPY_FINLINE npyv_f64 npyv_muladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return __riscv_vfmadd_vv_f64m1(a, b, c, npyv_nlanes_f64); }
NPY_FINLINE npyv_f64 npyv_mulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return __riscv_vfmsub_vv_f64m1(a, b, c, npyv_nlanes_f64); }
NPY_FINLINE npyv_f64 npyv_nmuladd_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return __riscv_vfnmsub_vv_f64m1(a, b, c, npyv_nlanes_f64); }
NPY_FINLINE npyv_f64 npyv_nmulsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return __riscv_vfnmadd_vv_f64m1(a, b, c, npyv_nlanes_f64); }

NPY_FINLINE npyv_f64 npyv_muladdsub_f64(npyv_f64 a, npyv_f64 b, npyv_f64 c)
{ return npyv_muladd_f64(a, b, __riscv_vfneg_v_f64m1_mu(__riscv_vreinterpret_v_u8m1_b64(__riscv_vmv_v_x_u8m1(0x55, npyv_nlanes_u8)), c, c, npyv_nlanes_f64)); }

/***************************
 * Summation
 ***************************/
// reduce sum across vector
NPY_FINLINE npy_uint32 npyv_sum_u32(npyv_u32 a)
{ return __riscv_vmv_x(__riscv_vredsum(a, __riscv_vmv_s_x_u32m1(0, 1), npyv_nlanes_u32)); }
NPY_FINLINE npy_uint64 npyv_sum_u64(npyv_u64 a)
{ return __riscv_vmv_x(__riscv_vredsum(a, __riscv_vmv_s_x_u64m1(0, 1), npyv_nlanes_u64)); }
NPY_FINLINE float npyv_sum_f32(npyv_f32 a)
{ return __riscv_vfmv_f(__riscv_vfredosum(a, __riscv_vfmv_s_f_f32m1(0, 1), npyv_nlanes_f32)); }
NPY_FINLINE double npyv_sum_f64(npyv_f64 a)
{ return __riscv_vfmv_f(__riscv_vfredosum(a, __riscv_vfmv_s_f_f64m1(0, 1), npyv_nlanes_f64)); }

NPY_FINLINE npy_uint16 npyv_sumup_u8(npyv_u8 a)
{ return __riscv_vmv_x(__riscv_vwredsumu(a, __riscv_vmv_s_x_u16m1(0, 1), npyv_nlanes_u8)); }
NPY_FINLINE npy_uint32 npyv_sumup_u16(npyv_u16 a)
{ return __riscv_vmv_x(__riscv_vwredsumu(a, __riscv_vmv_s_x_u32m1(0, 1), npyv_nlanes_u16)); }

#endif // _NPY_SIMD_RVV_ARITHMETIC_H
