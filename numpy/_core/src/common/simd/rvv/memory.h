#ifndef NPY_SIMD
    #error "Not a standalone header"
#endif

#ifndef _NPY_SIMD_RVV_MEMORY_H
#define _NPY_SIMD_RVV_MEMORY_H

#include "misc.h"

/***************************
 * load/store
 ***************************/
// GCC requires literal type definitions for pointers types otherwise it causes ambiguous errors

// uint8_t
NPY_FINLINE npyv_u8 npyv_load_u8(const npyv_lanetype_u8 *ptr)
{ return __riscv_vle8_v_u8m1((const uint8_t*)ptr, npyv_nlanes_u8); }
NPY_FINLINE npyv_u8 npyv_loada_u8(const npyv_lanetype_u8 *ptr)
{ return __riscv_vle8_v_u8m1((const uint8_t*)ptr, npyv_nlanes_u8); }
NPY_FINLINE npyv_u8 npyv_loads_u8(const npyv_lanetype_u8 *ptr)
{ return __riscv_vle8_v_u8m1((const uint8_t*)ptr, npyv_nlanes_u8); }
NPY_FINLINE npyv_u8 npyv_loadl_u8(const npyv_lanetype_u8 *ptr)
{ return __riscv_vle8_v_u8m1_tu(__riscv_vmv_v_x_u8m1(0, npyv_nlanes_u8), (const uint8_t*)ptr, npyv_nlanes_u8 / 2); }

NPY_FINLINE void npyv_store_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, vec, npyv_nlanes_u8); }
NPY_FINLINE void npyv_storea_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, vec, npyv_nlanes_u8); }
NPY_FINLINE void npyv_stores_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, vec, npyv_nlanes_u8); }
NPY_FINLINE void npyv_storel_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, vec, npyv_nlanes_u8 / 2); }
NPY_FINLINE void npyv_storeh_u8(npyv_lanetype_u8 *ptr, npyv_u8 vec)
{ __riscv_vse8_v_u8m1((uint8_t*)ptr, __riscv_vslidedown_vx_u8m1(vec, npyv_nlanes_u8 / 2, npyv_nlanes_u8), npyv_nlanes_u8 / 2); }

// int8_t
NPY_FINLINE npyv_s8 npyv_load_s8(const npyv_lanetype_s8 *ptr)
{ return __riscv_vle8_v_i8m1((const int8_t*)ptr, npyv_nlanes_s8); }
NPY_FINLINE npyv_s8 npyv_loada_s8(const npyv_lanetype_s8 *ptr)
{ return __riscv_vle8_v_i8m1((const int8_t*)ptr, npyv_nlanes_s8); }
NPY_FINLINE npyv_s8 npyv_loads_s8(const npyv_lanetype_s8 *ptr)
{ return __riscv_vle8_v_i8m1((const int8_t*)ptr, npyv_nlanes_s8); }
NPY_FINLINE npyv_s8 npyv_loadl_s8(const npyv_lanetype_s8 *ptr)
{ return __riscv_vle8_v_i8m1_tu(__riscv_vmv_v_x_i8m1(0, npyv_nlanes_s8), (const int8_t*)ptr, npyv_nlanes_s8 / 2); }

NPY_FINLINE void npyv_store_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, vec, npyv_nlanes_s8); }
NPY_FINLINE void npyv_storea_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, vec, npyv_nlanes_s8); }
NPY_FINLINE void npyv_stores_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, vec, npyv_nlanes_s8); }
NPY_FINLINE void npyv_storel_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, vec, npyv_nlanes_s8 / 2); }
NPY_FINLINE void npyv_storeh_s8(npyv_lanetype_s8 *ptr, npyv_s8 vec)
{ __riscv_vse8_v_i8m1((int8_t*)ptr, __riscv_vslidedown_vx_i8m1(vec, npyv_nlanes_s8 / 2, npyv_nlanes_s8), npyv_nlanes_s8 / 2); }

// uint16_t
NPY_FINLINE npyv_u16 npyv_load_u16(const npyv_lanetype_u16 *ptr)
{ return __riscv_vle16_v_u16m1((const uint16_t*)ptr, npyv_nlanes_u16); }
NPY_FINLINE npyv_u16 npyv_loada_u16(const npyv_lanetype_u16 *ptr)
{ return __riscv_vle16_v_u16m1((const uint16_t*)ptr, npyv_nlanes_u16); }
NPY_FINLINE npyv_u16 npyv_loads_u16(const npyv_lanetype_u16 *ptr)
{ return __riscv_vle16_v_u16m1((const uint16_t*)ptr, npyv_nlanes_u16); }
NPY_FINLINE npyv_u16 npyv_loadl_u16(const npyv_lanetype_u16 *ptr)
{ return __riscv_vle16_v_u16m1_tu(__riscv_vmv_v_x_u16m1(0, npyv_nlanes_u16), (const uint16_t*)ptr, npyv_nlanes_u16 / 2); }

NPY_FINLINE void npyv_store_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, vec, npyv_nlanes_u16); }
NPY_FINLINE void npyv_storea_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, vec, npyv_nlanes_u16); }
NPY_FINLINE void npyv_stores_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, vec, npyv_nlanes_u16); }
NPY_FINLINE void npyv_storel_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, vec, npyv_nlanes_u16 / 2); }
NPY_FINLINE void npyv_storeh_u16(npyv_lanetype_u16 *ptr, npyv_u16 vec)
{ __riscv_vse16_v_u16m1((uint16_t*)ptr, __riscv_vslidedown_vx_u16m1(vec, npyv_nlanes_u16 / 2, npyv_nlanes_u16), npyv_nlanes_u16 / 2); }

// int16_t
NPY_FINLINE npyv_s16 npyv_load_s16(const npyv_lanetype_s16 *ptr)
{ return __riscv_vle16_v_i16m1((const int16_t*)ptr, npyv_nlanes_s16); }
NPY_FINLINE npyv_s16 npyv_loada_s16(const npyv_lanetype_s16 *ptr)
{ return __riscv_vle16_v_i16m1((const int16_t*)ptr, npyv_nlanes_s16); }
NPY_FINLINE npyv_s16 npyv_loads_s16(const npyv_lanetype_s16 *ptr)
{ return __riscv_vle16_v_i16m1((const int16_t*)ptr, npyv_nlanes_s16); }
NPY_FINLINE npyv_s16 npyv_loadl_s16(const npyv_lanetype_s16 *ptr)
{ return __riscv_vle16_v_i16m1_tu(__riscv_vmv_v_x_i16m1(0, npyv_nlanes_s16), (const int16_t*)ptr, npyv_nlanes_s16 / 2); }

NPY_FINLINE void npyv_store_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, vec, npyv_nlanes_s16); }
NPY_FINLINE void npyv_storea_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, vec, npyv_nlanes_s16); }
NPY_FINLINE void npyv_stores_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, vec, npyv_nlanes_s16); }
NPY_FINLINE void npyv_storel_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, vec, npyv_nlanes_s16 / 2); }
NPY_FINLINE void npyv_storeh_s16(npyv_lanetype_s16 *ptr, npyv_s16 vec)
{ __riscv_vse16_v_i16m1((int16_t*)ptr, __riscv_vslidedown_vx_i16m1(vec, npyv_nlanes_s16 / 2, npyv_nlanes_s16), npyv_nlanes_s16 / 2); }

// uint32_t
NPY_FINLINE npyv_u32 npyv_load_u32(const npyv_lanetype_u32 *ptr)
{ return __riscv_vle32_v_u32m1((const uint32_t*)ptr, npyv_nlanes_u32); }
NPY_FINLINE npyv_u32 npyv_loada_u32(const npyv_lanetype_u32 *ptr)
{ return __riscv_vle32_v_u32m1((const uint32_t*)ptr, npyv_nlanes_u32); }
NPY_FINLINE npyv_u32 npyv_loads_u32(const npyv_lanetype_u32 *ptr)
{ return __riscv_vle32_v_u32m1((const uint32_t*)ptr, npyv_nlanes_u32); }
NPY_FINLINE npyv_u32 npyv_loadl_u32(const npyv_lanetype_u32 *ptr)
{ return __riscv_vle32_v_u32m1_tu(__riscv_vmv_v_x_u32m1(0, npyv_nlanes_u32), (const uint32_t*)ptr, npyv_nlanes_u32 / 2); }

NPY_FINLINE void npyv_store_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, vec, npyv_nlanes_u32); }
NPY_FINLINE void npyv_storea_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, vec, npyv_nlanes_u32); }
NPY_FINLINE void npyv_stores_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, vec, npyv_nlanes_u32); }
NPY_FINLINE void npyv_storel_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, vec, npyv_nlanes_u32 / 2); }
NPY_FINLINE void npyv_storeh_u32(npyv_lanetype_u32 *ptr, npyv_u32 vec)
{ __riscv_vse32_v_u32m1((uint32_t*)ptr, __riscv_vslidedown_vx_u32m1(vec, npyv_nlanes_u32 / 2, npyv_nlanes_u32), npyv_nlanes_u32 / 2); }

// int32_t
NPY_FINLINE npyv_s32 npyv_load_s32(const npyv_lanetype_s32 *ptr)
{ return __riscv_vle32_v_i32m1((const int32_t*)ptr, npyv_nlanes_s32); }
NPY_FINLINE npyv_s32 npyv_loada_s32(const npyv_lanetype_s32 *ptr)
{ return __riscv_vle32_v_i32m1((const int32_t*)ptr, npyv_nlanes_s32); }
NPY_FINLINE npyv_s32 npyv_loads_s32(const npyv_lanetype_s32 *ptr)
{ return __riscv_vle32_v_i32m1((const int32_t*)ptr, npyv_nlanes_s32); }
NPY_FINLINE npyv_s32 npyv_loadl_s32(const npyv_lanetype_s32 *ptr)
{ return __riscv_vle32_v_i32m1_tu(__riscv_vmv_v_x_i32m1(0, npyv_nlanes_s32), (const int32_t*)ptr, npyv_nlanes_s32 / 2); }

NPY_FINLINE void npyv_store_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, vec, npyv_nlanes_s32); }
NPY_FINLINE void npyv_storea_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, vec, npyv_nlanes_s32); }
NPY_FINLINE void npyv_stores_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, vec, npyv_nlanes_s32); }
NPY_FINLINE void npyv_storel_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, vec, npyv_nlanes_s32 / 2); }
NPY_FINLINE void npyv_storeh_s32(npyv_lanetype_s32 *ptr, npyv_s32 vec)
{ __riscv_vse32_v_i32m1((int32_t*)ptr, __riscv_vslidedown_vx_i32m1(vec, npyv_nlanes_s32 / 2, npyv_nlanes_s32), npyv_nlanes_s32 / 2); }

// uint64_t
NPY_FINLINE npyv_u64 npyv_load_u64(const npyv_lanetype_u64 *ptr)
{ return __riscv_vle64_v_u64m1((const uint64_t*)ptr, npyv_nlanes_u64); }
NPY_FINLINE npyv_u64 npyv_loada_u64(const npyv_lanetype_u64 *ptr)
{ return __riscv_vle64_v_u64m1((const uint64_t*)ptr, npyv_nlanes_u64); }
NPY_FINLINE npyv_u64 npyv_loads_u64(const npyv_lanetype_u64 *ptr)
{ return __riscv_vle64_v_u64m1((const uint64_t*)ptr, npyv_nlanes_u64); }
NPY_FINLINE npyv_u64 npyv_loadl_u64(const npyv_lanetype_u64 *ptr)
{ return __riscv_vle64_v_u64m1_tu(__riscv_vmv_v_x_u64m1(0, npyv_nlanes_u64), (const uint64_t*)ptr, npyv_nlanes_u64 / 2); }

NPY_FINLINE void npyv_store_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, vec, npyv_nlanes_u64); }
NPY_FINLINE void npyv_storea_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, vec, npyv_nlanes_u64); }
NPY_FINLINE void npyv_stores_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, vec, npyv_nlanes_u64); }
NPY_FINLINE void npyv_storel_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, vec, npyv_nlanes_u64 / 2); }
NPY_FINLINE void npyv_storeh_u64(npyv_lanetype_u64 *ptr, npyv_u64 vec)
{ __riscv_vse64_v_u64m1((uint64_t*)ptr, __riscv_vslidedown_vx_u64m1(vec, npyv_nlanes_u64 / 2, npyv_nlanes_u64), npyv_nlanes_u64 / 2); }

// int64_t
NPY_FINLINE npyv_s64 npyv_load_s64(const npyv_lanetype_s64 *ptr)
{ return __riscv_vle64_v_i64m1((const int64_t*)ptr, npyv_nlanes_s64); }
NPY_FINLINE npyv_s64 npyv_loada_s64(const npyv_lanetype_s64 *ptr)
{ return __riscv_vle64_v_i64m1((const int64_t*)ptr, npyv_nlanes_s64); }
NPY_FINLINE npyv_s64 npyv_loads_s64(const npyv_lanetype_s64 *ptr)
{ return __riscv_vle64_v_i64m1((const int64_t*)ptr, npyv_nlanes_s64); }
NPY_FINLINE npyv_s64 npyv_loadl_s64(const npyv_lanetype_s64 *ptr)
{ return __riscv_vle64_v_i64m1_tu(__riscv_vmv_v_x_i64m1(0, npyv_nlanes_s64), (const int64_t*)ptr, npyv_nlanes_s64 / 2); }

NPY_FINLINE void npyv_store_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, vec, npyv_nlanes_s64); }
NPY_FINLINE void npyv_storea_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, vec, npyv_nlanes_s64); }
NPY_FINLINE void npyv_stores_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, vec, npyv_nlanes_s64); }
NPY_FINLINE void npyv_storel_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, vec, npyv_nlanes_s64 / 2); }
NPY_FINLINE void npyv_storeh_s64(npyv_lanetype_s64 *ptr, npyv_s64 vec)
{ __riscv_vse64_v_i64m1((int64_t*)ptr, __riscv_vslidedown_vx_i64m1(vec, npyv_nlanes_s64 / 2, npyv_nlanes_s64), npyv_nlanes_s64 / 2); }

// float
NPY_FINLINE npyv_f32 npyv_load_f32(const npyv_lanetype_f32 *ptr)
{ return __riscv_vle32_v_f32m1((const float*)ptr, npyv_nlanes_f32); }
NPY_FINLINE npyv_f32 npyv_loada_f32(const npyv_lanetype_f32 *ptr)
{ return __riscv_vle32_v_f32m1((const float*)ptr, npyv_nlanes_f32); }
NPY_FINLINE npyv_f32 npyv_loads_f32(const npyv_lanetype_f32 *ptr)
{ return __riscv_vle32_v_f32m1((const float*)ptr, npyv_nlanes_f32); }
NPY_FINLINE npyv_f32 npyv_loadl_f32(const npyv_lanetype_f32 *ptr)
{ return __riscv_vle32_v_f32m1_tu(__riscv_vfmv_v_f_f32m1(0.0f, npyv_nlanes_f32), (const float*)ptr, npyv_nlanes_f32 / 2); }

NPY_FINLINE void npyv_store_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, vec, npyv_nlanes_f32); }
NPY_FINLINE void npyv_storea_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, vec, npyv_nlanes_f32); }
NPY_FINLINE void npyv_stores_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, vec, npyv_nlanes_f32); }
NPY_FINLINE void npyv_storel_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, vec, npyv_nlanes_f32 / 2); }
NPY_FINLINE void npyv_storeh_f32(npyv_lanetype_f32 *ptr, npyv_f32 vec)
{ __riscv_vse32_v_f32m1((float*)ptr, __riscv_vslidedown_vx_f32m1(vec, npyv_nlanes_f32 / 2, npyv_nlanes_f32), npyv_nlanes_f32 / 2); }

// double
NPY_FINLINE npyv_f64 npyv_load_f64(const npyv_lanetype_f64 *ptr)
{ return __riscv_vle64_v_f64m1((const double*)ptr, npyv_nlanes_f64); }
NPY_FINLINE npyv_f64 npyv_loada_f64(const npyv_lanetype_f64 *ptr)
{ return __riscv_vle64_v_f64m1((const double*)ptr, npyv_nlanes_f64); }
NPY_FINLINE npyv_f64 npyv_loads_f64(const npyv_lanetype_f64 *ptr)
{ return __riscv_vle64_v_f64m1((const double*)ptr, npyv_nlanes_f64); }
NPY_FINLINE npyv_f64 npyv_loadl_f64(const npyv_lanetype_f64 *ptr)
{ return __riscv_vle64_v_f64m1_tu(__riscv_vfmv_v_f_f64m1(0.0, npyv_nlanes_f64), (const double*)ptr, npyv_nlanes_f64 / 2); }

NPY_FINLINE void npyv_store_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, vec, npyv_nlanes_f64); }
NPY_FINLINE void npyv_storea_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, vec, npyv_nlanes_f64); }
NPY_FINLINE void npyv_stores_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, vec, npyv_nlanes_f64); }
NPY_FINLINE void npyv_storel_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, vec, npyv_nlanes_f64 / 2); }
NPY_FINLINE void npyv_storeh_f64(npyv_lanetype_f64 *ptr, npyv_f64 vec)
{ __riscv_vse64_v_f64m1((double*)ptr, __riscv_vslidedown_vx_f64m1(vec, npyv_nlanes_f64 / 2, npyv_nlanes_f64), npyv_nlanes_f64 / 2); }


/***************************
 * Non-contiguous Load
 ***************************/
NPY_FINLINE npyv_s32 npyv_loadn_s32(const npy_int32 *ptr, npy_intp stride)
{ return __riscv_vlse32_v_i32m1((const int32_t*)ptr, stride * sizeof(int32_t), npyv_nlanes_s32); }
NPY_FINLINE npyv_u32 npyv_loadn_u32(const npy_uint32 *ptr, npy_intp stride)
{ return __riscv_vlse32_v_u32m1((const uint32_t*)ptr, stride * sizeof(uint32_t), npyv_nlanes_u32); }
NPY_FINLINE npyv_f32 npyv_loadn_f32(const float *ptr, npy_intp stride)
{ return __riscv_vlse32_v_f32m1((const float*)ptr, stride * sizeof(float), npyv_nlanes_f32); }

NPY_FINLINE npyv_s64 npyv_loadn_s64(const npy_int64 *ptr, npy_intp stride)
{ return __riscv_vlse64_v_i64m1((const int64_t*)ptr, stride * sizeof(int64_t), npyv_nlanes_s64); }
NPY_FINLINE npyv_u64 npyv_loadn_u64(const npy_uint64 *ptr, npy_intp stride)
{ return __riscv_vlse64_v_u64m1((const uint64_t*)ptr, stride * sizeof(uint64_t), npyv_nlanes_u64); }
NPY_FINLINE npyv_f64 npyv_loadn_f64(const double *ptr, npy_intp stride)
{ return __riscv_vlse64_v_f64m1((const double*)ptr, stride * sizeof(double), npyv_nlanes_f64); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_u32 npyv_loadn2_u32(const npy_uint32 *ptr, npy_intp stride)
{ return __riscv_vreinterpret_v_u64m1_u32m1(__riscv_vlse64_v_u64m1((const uint64_t*)ptr, stride * sizeof(uint32_t), npyv_nlanes_u64)); }
NPY_FINLINE npyv_s32 npyv_loadn2_s32(const npy_int32 *ptr, npy_intp stride)
{ return npyv_reinterpret_s32_u32(npyv_loadn2_u32((const npy_uint32*)ptr, stride)); }
NPY_FINLINE npyv_f32 npyv_loadn2_f32(const float *ptr, npy_intp stride)
{ return npyv_reinterpret_f32_u32(npyv_loadn2_u32((const npy_uint32*)ptr, stride)); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_u64 npyv_loadn2_u64(const npy_uint64 *ptr, npy_intp stride)
{
    vuint64m1_t id = __riscv_vmul(__riscv_vsrl(__riscv_vid_v_u64m1(npyv_nlanes_u64), 1, npyv_nlanes_u64), stride * sizeof(uint64_t), npyv_nlanes_u64);
    id = __riscv_vadd_vx_u64m1_mu(__riscv_vreinterpret_v_u8m1_b64(__riscv_vmv_v_x_u8m1(0xAA, npyv_nlanes_u8)), id, id, sizeof(uint64_t), npyv_nlanes_u64);
    return __riscv_vloxei64_v_u64m1((const uint64_t*)ptr, id, npyv_nlanes_u64);
}
NPY_FINLINE npyv_s64 npyv_loadn2_s64(const npy_int64 *ptr, npy_intp stride)
{ return npyv_reinterpret_s64_u64(npyv_loadn2_u64((const npy_uint64*)ptr, stride)); }

NPY_FINLINE npyv_f64 npyv_loadn2_f64(const double *ptr, npy_intp stride)
{ return npyv_reinterpret_f64_u64(npyv_loadn2_u64((const npy_uint64*)ptr, stride)); }

/***************************
 * Non-contiguous Store
 ***************************/
NPY_FINLINE void npyv_storen_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ __riscv_vsse32((int32_t*)ptr, stride * sizeof(int32_t), a, npyv_nlanes_s32); }
NPY_FINLINE void npyv_storen_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ __riscv_vsse32((uint32_t*)ptr, stride * sizeof(uint32_t), a, npyv_nlanes_u32); }
NPY_FINLINE void npyv_storen_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ __riscv_vsse32((float*)ptr, stride * sizeof(float), a, npyv_nlanes_f32); }

NPY_FINLINE void npyv_storen_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ __riscv_vsse64((int64_t*)ptr, stride * sizeof(int64_t), a, npyv_nlanes_s64); }
NPY_FINLINE void npyv_storen_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{ __riscv_vsse64((uint64_t*)ptr, stride * sizeof(uint64_t), a, npyv_nlanes_u64); }
NPY_FINLINE void npyv_storen_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ __riscv_vsse64((double*)ptr, stride * sizeof(double), a, npyv_nlanes_f64); }

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_u32(npy_uint32 *ptr, npy_intp stride, npyv_u32 a)
{ __riscv_vsse64((uint64_t*)ptr, stride * sizeof(uint32_t), __riscv_vreinterpret_v_u32m1_u64m1(a), npyv_nlanes_u64); }
NPY_FINLINE void npyv_storen2_s32(npy_int32 *ptr, npy_intp stride, npyv_s32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, npyv_reinterpret_u32_s32(a)); }
NPY_FINLINE void npyv_storen2_f32(float *ptr, npy_intp stride, npyv_f32 a)
{ npyv_storen2_u32((npy_uint32*)ptr, stride, npyv_reinterpret_u32_f32(a)); }

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_u64(npy_uint64 *ptr, npy_intp stride, npyv_u64 a)
{
    vuint64m1_t id = __riscv_vmul(__riscv_vsrl(__riscv_vid_v_u64m1(npyv_nlanes_u64), 1, npyv_nlanes_u64), stride * sizeof(uint64_t), npyv_nlanes_u64);
    id = __riscv_vadd_vx_u64m1_mu(__riscv_vreinterpret_v_u8m1_b64(__riscv_vmv_v_x_u8m1(0xAA, npyv_nlanes_u8)), id, id, sizeof(uint64_t), npyv_nlanes_u64);
    __riscv_vsoxei64((uint64_t*)ptr, id, a, npyv_nlanes_u64);
}
NPY_FINLINE void npyv_storen2_s64(npy_int64 *ptr, npy_intp stride, npyv_s64 a)
{ npyv_storen2_u64((npy_uint64*)ptr, stride, npyv_reinterpret_u64_s64(a)); }
NPY_FINLINE void npyv_storen2_f64(double *ptr, npy_intp stride, npyv_f64 a)
{ npyv_storen2_u64((npy_uint64*)ptr, stride, npyv_reinterpret_u64_f64(a)); }

/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE npyv_s32 npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{ return __riscv_vle32_v_i32m1_tu(__riscv_vmv_v_x_i32m1(fill, npyv_nlanes_s32), (const int32_t*)ptr, nlane); }
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return npyv_load_till_s32(ptr, nlane, 0); }

NPY_FINLINE npyv_s64 npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{ return __riscv_vle64_v_i64m1_tu(__riscv_vmv_v_x_i64m1(fill, npyv_nlanes_s64), (const int64_t*)ptr, nlane); }
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ return npyv_load_till_s64(ptr, nlane, 0); }

//// 64-bit nlane
NPY_FINLINE npyv_s32 npyv_load2_till_s32(const npy_int32 *ptr, npy_uintp nlane,
                                          npy_int32 fill_lo, npy_int32 fill_hi)
{ return __riscv_vreinterpret_v_i64m1_i32m1(npyv_load_till_s64((const npy_int64*)ptr, nlane, (uint64_t)fill_hi << 32 | fill_lo)); }
// fill zero to rest lanes
NPY_FINLINE npyv_s32 npyv_load2_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{ return __riscv_vreinterpret_v_i64m1_i32m1(npyv_load_tillz_s64((const npy_int64*)ptr, nlane)); }

//// 128-bit nlane
NPY_FINLINE npyv_s64 npyv_load2_till_s64(const npy_int64 *ptr, npy_uintp nlane,
                                           npy_int64 fill_lo, npy_int64 fill_hi)
{
    const vint64m1_t fill = __riscv_vmerge(__riscv_vmv_v_x_i64m1(fill_lo, npyv_nlanes_s64), fill_hi, __riscv_vreinterpret_v_u8m1_b64(__riscv_vmv_v_x_u8m1(0xAA, npyv_nlanes_u8)), npyv_nlanes_s64);
    return __riscv_vle64_v_i64m1_tu(fill, (const int64_t*)ptr, nlane * 2);
}
NPY_FINLINE npyv_s64 npyv_load2_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{ return __riscv_vle64_v_i64m1_tu(__riscv_vmv_v_x_i64m1(0, npyv_nlanes_s64), (const int64_t*)ptr, nlane * 2); }

/*********************************
 * Non-contiguous partial load
 *********************************/
NPY_FINLINE npyv_s32 npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npy_int32 fill)
{ return __riscv_vlse32_v_i32m1_tu(__riscv_vmv_v_x_i32m1(fill, npyv_nlanes_s32), (const int32_t*)ptr, stride * sizeof(int32_t), nlane); }
NPY_FINLINE npyv_s32 npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s32(ptr, stride, nlane, 0); }

NPY_FINLINE npyv_s64 npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npy_int64 fill)
{ return __riscv_vlse64_v_i64m1_tu(__riscv_vmv_v_x_i64m1(fill, npyv_nlanes_s64), (const int64_t*)ptr, stride * sizeof(int64_t), nlane); }
// fill zero to rest lanes
NPY_FINLINE npyv_s64 npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn_till_s64(ptr, stride, nlane, 0); }

//// 64-bit load over 32-bit stride
NPY_FINLINE npyv_s32 npyv_loadn2_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                                                 npy_int32 fill_lo, npy_int32 fill_hi)
{ return __riscv_vreinterpret_v_i64m1_i32m1(__riscv_vlse64_v_i64m1_tu(__riscv_vmv_v_x_i64m1((uint64_t)fill_hi << 32 | fill_lo, npyv_nlanes_s64), (const int64_t*)ptr, stride * sizeof(int32_t), nlane)); }
NPY_FINLINE npyv_s32 npyv_loadn2_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn2_till_s32(ptr, stride, nlane, 0, 0); }

//// 128-bit load over 64-bit stride
NPY_FINLINE npyv_s64 npyv_loadn2_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                                          npy_int64 fill_lo, npy_int64 fill_hi)
{
    vbool64_t mask = __riscv_vreinterpret_v_u8m1_b64(__riscv_vmv_v_x_u8m1(0xAA, npyv_nlanes_u8));
    vint64m1_t fill = __riscv_vmerge(__riscv_vmv_v_x_i64m1(fill_lo, npyv_nlanes_s64), fill_hi, mask, npyv_nlanes_s64);
    vuint64m1_t id = __riscv_vmul(__riscv_vsrl(__riscv_vid_v_u64m1(npyv_nlanes_u64), 1, npyv_nlanes_u64), stride * sizeof(uint64_t), npyv_nlanes_u64);
    id = __riscv_vadd_vx_u64m1_mu(mask, id, id, sizeof(uint64_t), npyv_nlanes_u64);
    return __riscv_vloxei64_v_i64m1_tu(fill, (const int64_t*)ptr, id, nlane * 2);
}
NPY_FINLINE npyv_s64 npyv_loadn2_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{ return npyv_loadn2_till_s64(ptr, stride, nlane, 0, 0); }

/*********************************
 * Partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{ __riscv_vse32((int32_t*)ptr, a, nlane); }

//// 64
NPY_FINLINE void npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{ __riscv_vse64((int64_t*)ptr, a, nlane); }

//// 64-bit nlane
NPY_FINLINE void npyv_store2_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{ npyv_store_till_s64((npy_int64*)ptr, nlane, npyv_reinterpret_s64_s32(a)); }

//// 128-bit nlane
NPY_FINLINE void npyv_store2_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{ npyv_store_till_s64((npy_int64*)ptr, nlane * 2, a); }

/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{ __riscv_vsse32((int32_t*)ptr, stride * sizeof(int32_t), a, nlane); }

//// 64
NPY_FINLINE void npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{ __riscv_vsse64((int64_t*)ptr, stride * sizeof(int64_t), a, nlane); }

//// 64-bit store over 32-bit stride
NPY_FINLINE void npyv_storen2_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane, npyv_s32 a)
{ __riscv_vsse64((int64_t*)ptr, stride * sizeof(int32_t), __riscv_vreinterpret_v_i32m1_i64m1(a), nlane); }

//// 128-bit store over 64-bit stride
NPY_FINLINE void npyv_storen2_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane, npyv_s64 a)
{
    vuint64m1_t id = __riscv_vmul(__riscv_vsrl(__riscv_vid_v_u64m1(npyv_nlanes_u64), 1, npyv_nlanes_u64), stride * sizeof(uint64_t), npyv_nlanes_u64);
    id = __riscv_vadd_vx_u64m1_mu(__riscv_vreinterpret_v_u8m1_b64(__riscv_vmv_v_x_u8m1(0xAA, npyv_nlanes_u8)), id, id, sizeof(uint64_t), npyv_nlanes_u64);
    __riscv_vsoxei64((int64_t*)ptr, id, a, nlane * 2);
}

/*****************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via casting
 *****************************************************************/
#define NPYV_IMPL_RVV_REST_PARTIAL_TYPES(F_SFX, T_SFX)                                      \
    NPY_FINLINE npyv_##F_SFX npyv_load_till_##F_SFX                                         \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_lanetype_##F_SFX fill)         \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(                   \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill)                                                            \
    {                                                                                       \
        union {                                                                             \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        } pun;                                                                              \
        pun.from_##F_SFX = fill;                                                            \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun.to_##T_SFX               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE void npyv_store_till_##F_SFX                                                \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        npyv_store_till_##T_SFX(                                                            \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }

NPYV_IMPL_RVV_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES(f64, s64)
#undef NPYV_IMPL_RVV_REST_PARTIAL_TYPES

// 128-bit/64-bit stride
#define NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(F_SFX, T_SFX)                                 \
    NPY_FINLINE npyv_##F_SFX npyv_load2_till_##F_SFX                                        \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                                     \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_till_##T_SFX(                  \
            (const npyv_lanetype_##T_SFX *)ptr, nlane, pun_lo.to_##T_SFX, pun_hi.to_##T_SFX \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_till_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,                    \
     npyv_lanetype_##F_SFX fill_lo, npyv_lanetype_##F_SFX fill_hi)                          \
    {                                                                                       \
        union pun {                                                                         \
            npyv_lanetype_##F_SFX from_##F_SFX;                                             \
            npyv_lanetype_##T_SFX to_##T_SFX;                                               \
        };                                                                                  \
        union pun pun_lo;                                                                   \
        union pun pun_hi;                                                                   \
        pun_lo.from_##F_SFX = fill_lo;                                                      \
        pun_hi.from_##F_SFX = fill_hi;                                                      \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_till_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane, pun_lo.to_##T_SFX,           \
            pun_hi.to_##T_SFX                                                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_load2_tillz_##F_SFX                                       \
    (const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                                     \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load2_tillz_##T_SFX(                 \
            (const npyv_lanetype_##T_SFX *)ptr, nlane                                       \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE npyv_##F_SFX npyv_loadn2_tillz_##F_SFX                                      \
    (const npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane)                    \
    {                                                                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn2_tillz_##T_SFX(                \
            (const npyv_lanetype_##T_SFX *)ptr, stride, nlane                               \
        ));                                                                                 \
    }                                                                                       \
    NPY_FINLINE void npyv_store2_till_##F_SFX                                               \
    (npyv_lanetype_##F_SFX *ptr, npy_uintp nlane, npyv_##F_SFX a)                           \
    {                                                                                       \
        npyv_store2_till_##T_SFX(                                                           \
            (npyv_lanetype_##T_SFX *)ptr, nlane,                                            \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }                                                                                       \
    NPY_FINLINE void npyv_storen2_till_##F_SFX                                              \
    (npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane, npyv_##F_SFX a)          \
    {                                                                                       \
        npyv_storen2_till_##T_SFX(                                                          \
            (npyv_lanetype_##T_SFX *)ptr, stride, nlane,                                    \
            npyv_reinterpret_##T_SFX##_##F_SFX(a)                                           \
        );                                                                                  \
    }

NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(u32, s32)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(f32, s32)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(u64, s64)
NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR(f64, s64)
#undef NPYV_IMPL_RVV_REST_PARTIAL_TYPES_PAIR

/************************************************************
 *  de-interleave load / interleave contiguous store
 ************************************************************/
// two channels
#define NPYV_IMPL_RVV_MEM_INTERLEAVE(SFX, R_SFX, EEW)              \
    NPY_FINLINE npyv_##SFX##x2 npyv_load_##SFX##x2(                \
        const npyv_lanetype_##SFX *ptr                             \
    ) {                                                            \
        npyv__##SFX##x2 v = __riscv_vlseg2##EEW##_v_##R_SFX##m1x2( \
            ptr, npyv_nlanes_##SFX                                 \
        );                                                         \
        return (npyv_##SFX##x2){{                                  \
            __riscv_vget_v_##R_SFX##m1x2_##R_SFX##m1(v, 0),        \
            __riscv_vget_v_##R_SFX##m1x2_##R_SFX##m1(v, 1)         \
        }};                                                        \
    }                                                              \
    NPY_FINLINE void npyv_store_##SFX##x2(                         \
        npyv_lanetype_##SFX *ptr, npyv_##SFX##x2 v                 \
    ) {                                                            \
        __riscv_vsseg2##EEW(                                       \
            ptr,                                                   \
            __riscv_vcreate_v_##R_SFX##m1x2(v.val[0], v.val[1]),   \
            npyv_nlanes_##SFX                                      \
        );                                                         \
    }

NPYV_IMPL_RVV_MEM_INTERLEAVE(u8, u8, e8)
NPYV_IMPL_RVV_MEM_INTERLEAVE(s8, i8, e8)
NPYV_IMPL_RVV_MEM_INTERLEAVE(u16, u16, e16)
NPYV_IMPL_RVV_MEM_INTERLEAVE(s16, i16, e16)
NPYV_IMPL_RVV_MEM_INTERLEAVE(u32, u32, e32)
NPYV_IMPL_RVV_MEM_INTERLEAVE(s32, i32, e32)
NPYV_IMPL_RVV_MEM_INTERLEAVE(u64, u64, e64)
NPYV_IMPL_RVV_MEM_INTERLEAVE(s64, i64, e64)
NPYV_IMPL_RVV_MEM_INTERLEAVE(f32, f32, e32)
NPYV_IMPL_RVV_MEM_INTERLEAVE(f64, f64, e64)
#undef NPYV_IMPL_RVV_MEM_INTERLEAVE

/*********************************
 * Lookup table
 *********************************/
// uses vector as indexes into a table
// that contains 32 elements of uint32.
NPY_FINLINE npyv_u32 npyv_lut32_u32(const npy_uint32 *table, npyv_u32 idx)
{ return __riscv_vloxei32_v_u32m1((const uint32_t*)table, __riscv_vmul(idx, sizeof(uint32_t), npyv_nlanes_u32), npyv_nlanes_u32); }
NPY_FINLINE npyv_s32 npyv_lut32_s32(const npy_int32 *table, npyv_u32 idx)
{ return npyv_reinterpret_s32_u32(npyv_lut32_u32((const npy_uint32*)table, idx)); }
NPY_FINLINE npyv_f32 npyv_lut32_f32(const float *table, npyv_u32 idx)
{ return npyv_reinterpret_f32_u32(npyv_lut32_u32((const npy_uint32*)table, idx)); }

// uses vector as indexes into a table
// that contains 16 elements of uint64.
NPY_FINLINE npyv_u64 npyv_lut16_u64(const npy_uint64 *table, npyv_u64 idx)
{ return __riscv_vloxei64_v_u64m1((const uint64_t*)table, __riscv_vmul(idx, sizeof(uint64_t), npyv_nlanes_u64), npyv_nlanes_u64); }
NPY_FINLINE npyv_s64 npyv_lut16_s64(const npy_int64 *table, npyv_u64 idx)
{ return npyv_reinterpret_s64_u64(npyv_lut16_u64((const npy_uint64*)table, idx)); }
NPY_FINLINE npyv_f64 npyv_lut16_f64(const double *table, npyv_u64 idx)
{ return npyv_reinterpret_f64_u64(npyv_lut16_u64((const npy_uint64*)table, idx)); }

#endif // _NPY_SIMD_RVV_MEMORY_H
