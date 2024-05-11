#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"
#include "numpy/utils.h"
#include "fast_loop_macros.h"
#include "loops_utils.h"
#include <hwy/highway.h>
#include <hwy/aligned_allocator.h>

namespace hn = hwy::HWY_NAMESPACE;

enum HWY_OP {
  HWY_ROUND,
  HWY_FLOOR,
  HWY_CEIL,
  HWY_TRUNC,
  HWY_SQRT,
  HWY_SQUARE,
  HWY_ABS,
  HWY_RECIPROCAL,
};

#define FUNC(result, in)                        \
  switch (op) {                                 \
    case HWY_ROUND:                             \
      result = hn::Round(in);                   \
      break;                                    \
    case HWY_FLOOR:                             \
      result = hn::Floor(in);                   \
      break;                                    \
    case HWY_CEIL:                              \
      result = hn::Ceil(in);                    \
      break;                                    \
    case HWY_TRUNC:                             \
      result = hn::Trunc(in);                   \
      break;                                    \
    case HWY_SQRT:                              \
      result = hn::Sqrt(in);                    \
      break;                                    \
    case HWY_SQUARE:                            \
      result = hn::Mul(in, in);                 \
      break;                                    \
    case HWY_ABS:                               \
      result = hn::Abs(in);                     \
      break;                                    \
    case HWY_RECIPROCAL:                        \
      result = hn::Div(hn::Set(d, T(1.0)), in); \
      break;                                    \
  }

template <typename T>
HWY_ATTR void Super(char** args,
                    npy_intp const* dimensions,
                    npy_intp const* steps,
                    HWY_OP op,
                    bool IS_RECIP) {
  const T* HWY_RESTRICT input_array = (const T*)args[0];
  T* HWY_RESTRICT output_array = (T*)args[1];
  const size_t size = dimensions[0];
  const hn::ScalableTag<T> d;
  if (is_mem_overlap(input_array, steps[0], output_array, steps[1], size)) {
    const int lsize = sizeof(input_array[0]);
    const npy_intp ssrc = steps[0] / lsize;
    const npy_intp sdst = steps[1] / lsize;
    for (size_t len = size; 0 < len;
         len--, input_array += ssrc, output_array += sdst) {
      const auto in = hn::LoadN(d, input_array, 1);
      hn::VFromD<hn::ScalableTag<T>> x;
      FUNC(x, in);
      hn::StoreN(x, d, output_array, 1);
    }
  } else if (IS_UNARY_CONT(input_array[0], output_array[0])) {
    const int vstep = hn::Lanes(d);
    const int wstep = vstep * 4;
    size_t len = size;
    for (; len >= wstep;
         len -= wstep, input_array += wstep, output_array += wstep) {
      const auto in0 = hn::LoadU(d, input_array + vstep * 0);
      hn::VFromD<hn::ScalableTag<T>> x0;
      FUNC(x0, in0);
      const auto in1 = hn::LoadU(d, input_array + vstep * 1);
      hn::VFromD<hn::ScalableTag<T>> x1;
      FUNC(x1, in1);
      const auto in2 = hn::LoadU(d, input_array + vstep * 2);
      hn::VFromD<hn::ScalableTag<T>> x2;
      FUNC(x2, in2);
      const auto in3 = hn::LoadU(d, input_array + vstep * 3);
      hn::VFromD<hn::ScalableTag<T>> x3;
      FUNC(x3, in3);
      hn::StoreU(x0, d, output_array + vstep * 0);
      hn::StoreU(x1, d, output_array + vstep * 1);
      hn::StoreU(x2, d, output_array + vstep * 2);
      hn::StoreU(x3, d, output_array + vstep * 3);
    }
    for (; len >= vstep;
         len -= vstep, input_array += vstep, output_array += vstep) {
      const auto in = hn::LoadU(d, input_array);
      hn::VFromD<hn::ScalableTag<T>> x;
      FUNC(x, in);
      hn::StoreU(x, d, output_array);
    }
    if (len) {
      hn::Vec<hn::ScalableTag<T>> in;
      if (IS_RECIP) {
        auto one = hn::Set(d, 1);
        in = hn::LoadNOr(one, d, input_array, len);
      } else {
        in = hn::LoadN(d, input_array, len);
      }
      hn::VFromD<hn::ScalableTag<T>> x;
      FUNC(x, in);
      hn::StoreN(x, d, output_array, len);
    }
  } else {
    using TI = hwy::MakeSigned<T>;
    const hn::Rebind<TI, hn::ScalableTag<T>> di;

    const int lsize = sizeof(input_array[0]);
    const npy_intp ssrc = steps[0] / lsize;
    const npy_intp sdst = steps[1] / lsize;
    auto load_index = hn::Mul(hn::Iota(di, 0), hn::Set(di, ssrc));
    auto store_index = hn::Mul(hn::Iota(di, 0), hn::Set(di, sdst));
    size_t full = size & -hn::Lanes(d);
    size_t remainder = size - full;
    for (size_t i = 0; i < full; i += hn::Lanes(d)) {
      const auto in = hn::GatherIndex(d, input_array + i * ssrc, load_index);
      hn::VFromD<hn::ScalableTag<T>> x;
      FUNC(x, in);
      hn::ScatterIndex(x, d, output_array + i * sdst, store_index);
    }
    if (remainder) {
      hn::Vec<hn::ScalableTag<T>> in;
      if (IS_RECIP) {
        auto one = hn::Set(d, 1);
        in = hn::GatherIndexNOr(one, d, input_array + full * ssrc, load_index,
                                remainder);
      } else {
        in = hn::GatherIndexN(d, input_array + full * ssrc, load_index,
                              remainder);
      }
      hn::VFromD<hn::ScalableTag<T>> x;
      FUNC(x, in);
      hn::ScatterIndexN(x, d, output_array + full * sdst, store_index,
                        remainder);
    }
  }
}

extern "C" {
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double>(args, dimensions, steps, HWY_OP::HWY_ROUND, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float>(args, dimensions, steps, HWY_OP::HWY_ROUND, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double>(args, dimensions, steps, HWY_OP::HWY_FLOOR, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float>(args, dimensions, steps, HWY_OP::HWY_FLOOR, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double>(args, dimensions, steps, HWY_OP::HWY_CEIL, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float>(args, dimensions, steps, HWY_OP::HWY_CEIL, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double>(args, dimensions, steps, HWY_OP::HWY_TRUNC, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float>(args, dimensions, steps, HWY_OP::HWY_TRUNC, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_sqrt)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double>(args, dimensions, steps, HWY_OP::HWY_SQRT, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_sqrt)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float>(args, dimensions, steps, HWY_OP::HWY_SQRT, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_square)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_double>(args, dimensions, steps, HWY_OP::HWY_SQUARE, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_square)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return Super<npy_float>(args, dimensions, steps, HWY_OP::HWY_SQUARE, false);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  Super<npy_double>(args, dimensions, steps, HWY_OP::HWY_ABS, false);
  npy_clear_floatstatus_barrier((char*)dimensions);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  Super<npy_float>(args, dimensions, steps, HWY_OP::HWY_ABS, false);
  npy_clear_floatstatus_barrier((char*)dimensions);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  Super<npy_double>(args, dimensions, steps, HWY_OP::HWY_RECIPROCAL, true);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_reciprocal)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  Super<npy_float>(args, dimensions, steps, HWY_OP::HWY_RECIPROCAL, true);
}
}