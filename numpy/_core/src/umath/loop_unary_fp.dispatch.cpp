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

// Alternative to per-function HWY_ATTR: see HWY_BEFORE_NAMESPACE
#define SUPER(NAME, FUNC, IS_RECIP)                                            \
  template <typename T>                                                        \
  HWY_ATTR void Super##NAME(char** args, npy_intp const* dimensions,           \
                            npy_intp const* steps) {                           \
    const T* HWY_RESTRICT input_array = (const T*)args[0];                     \
    T* HWY_RESTRICT output_array = (T*)args[1];                                \
    const size_t size = dimensions[0];                                         \
    const hn::ScalableTag<T> d;                                                \
                                                                               \
    if (is_mem_overlap(input_array, steps[0], output_array, steps[1], size)) { \
      const int lsize = sizeof(input_array[0]);                                \
      const npy_intp ssrc = steps[0] / lsize;                                  \
      const npy_intp sdst = steps[1] / lsize;                                  \
      for (size_t len = size; 0 < len;                                         \
           len--, input_array += ssrc, output_array += sdst) {                 \
        const auto in = hn::LoadN(d, input_array, 1);                          \
        auto x = FUNC(in);                                                     \
        hn::StoreN(x, d, output_array, 1);                                     \
      }                                                                        \
    } else if (IS_UNARY_CONT(input_array, output_array)) {                     \
      const int vstep = hn::Lanes(d);                                          \
      const int wstep = vstep * 4;                                             \
      size_t len = size;                                                       \
      for (; len >= wstep;                                                     \
           len -= wstep, input_array += wstep, output_array += wstep) {        \
        const auto in0 = hn::LoadU(d, input_array + vstep * 0);                \
        auto x0 = FUNC(in0);                                                   \
        const auto in1 = hn::LoadU(d, input_array + vstep * 1);                \
        auto x1 = FUNC(in1);                                                   \
        const auto in2 = hn::LoadU(d, input_array + vstep * 2);                \
        auto x2 = FUNC(in2);                                                   \
        const auto in3 = hn::LoadU(d, input_array + vstep * 3);                \
        auto x3 = FUNC(in3);                                                   \
        hn::StoreU(x0, d, output_array + vstep * 0);                           \
        hn::StoreU(x1, d, output_array + vstep * 1);                           \
        hn::StoreU(x2, d, output_array + vstep * 2);                           \
        hn::StoreU(x3, d, output_array + vstep * 3);                           \
      }                                                                        \
      for (; len >= vstep;                                                     \
           len -= vstep, input_array += vstep, output_array += vstep) {        \
        const auto in = hn::LoadU(d, input_array);                             \
        auto x = FUNC(in);                                                     \
        hn::StoreU(x, d, output_array);                                        \
      }                                                                        \
      if (len) {                                                               \
        hn::Vec<hn::ScalableTag<T>> in;                                        \
        if (IS_RECIP) {                                                        \
          auto one = hn::Set(d, 1);                                            \
          in = hn::LoadNOr(one, d, input_array, len);                          \
        } else {                                                               \
          in = hn::LoadN(d, input_array, len);                                 \
        }                                                                      \
        auto x = FUNC(in);                                                     \
        hn::StoreN(x, d, output_array, len);                                   \
      }                                                                        \
    } else {                                                                   \
      using TI = hwy::MakeSigned<T>;                                           \
      const hn::Rebind<TI, hn::ScalableTag<T>> di;                             \
                                                                               \
      const int lsize = sizeof(input_array[0]);                                \
      const npy_intp ssrc = steps[0] / lsize;                                  \
      const npy_intp sdst = steps[1] / lsize;                                  \
      auto load_index = hn::Mul(hn::Iota(di, 0), hn::Set(di, ssrc));           \
      auto store_index = hn::Mul(hn::Iota(di, 0), hn::Set(di, sdst));          \
      size_t full = size & -hn::Lanes(d);                                      \
      size_t remainder = size - full;                                          \
      for (size_t i = 0; i < full; i += hn::Lanes(d)) {                        \
        const auto in =                                                        \
            hn::GatherIndex(d, input_array + i * ssrc, load_index);            \
        auto x = FUNC(in);                                                     \
        hn::ScatterIndex(x, d, output_array + i * sdst, store_index);          \
      }                                                                        \
      if (remainder) {                                                         \
        const auto in = hn::GatherIndexN(d, input_array + full * ssrc,         \
                                         load_index, remainder);               \
        auto x = FUNC(in);                                                     \
        hn::ScatterIndexN(x, d, output_array + full * sdst, store_index,       \
                          remainder);                                          \
      }                                                                        \
    }                                                                          \
  }

#define Square(x) hn::Mul(x, x) 
//ceil, trunc, sqrt, square, 
SUPER(Rint, hn::Round, false)
SUPER(Floor, hn::Floor, false)
SUPER(Ceil, hn::Ceil, false)
SUPER(Trunc, hn::Trunc, false)
SUPER(Sqrt, hn::Sqrt, false)
SUPER(Square, Square, false)
SUPER(Absolute, hn::Abs, false)

extern "C" {
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperRint<npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_rint)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperRint<npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperFloor<npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_floor)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperFloor<npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperCeil<npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_ceil)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperCeil<npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperTrunc<npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_trunc)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperTrunc<npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_sqrt)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperSqrt<npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_sqrt)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperSqrt<npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_square)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperSquare<npy_double>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_square)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  return SuperSquare<npy_float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(DOUBLE_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  SuperAbsolute<npy_double>(args, dimensions, steps);
  npy_clear_floatstatus_barrier((char*)dimensions);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(FLOAT_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  SuperAbsolute<npy_float>(args, dimensions, steps);
  npy_clear_floatstatus_barrier((char*)dimensions);
}
}