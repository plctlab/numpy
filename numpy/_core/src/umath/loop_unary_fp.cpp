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
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "loop_unary_fp.cpp"  // this file
#include <hwy/foreach_target.h>  // must come before highway.h
#include <hwy/highway.h>
#include <hwy/aligned_allocator.h>


namespace numpy {
namespace HWY_NAMESPACE {  // required: unique per target

// Can skip hn:: prefixes if already inside hwy::HWY_NAMESPACE.
namespace hn = hwy::HWY_NAMESPACE;

// Alternative to per-function HWY_ATTR: see HWY_BEFORE_NAMESPACE
#define SUPER(NAME, FUNC)                                                      \
  template <typename T>                                                        \
  HWY_ATTR void Super##NAME(char** args, npy_intp const* dimensions,           \
                          npy_intp const* steps) {                             \
    const T* HWY_RESTRICT input_array = (const T*)args[0];                     \
    T* HWY_RESTRICT output_array = (T*)args[1];                                \
    const size_t size = dimensions[0];                                         \
    const hn::ScalableTag<T> d;                                                \
                                                                               \
    if (is_mem_overlap(input_array, steps[0], output_array, steps[1], size)) { \
      for (size_t i = 0; i < size; i++) {                                      \
        const auto in = hn::LoadN(d, input_array + i, 1);                      \
        auto x = FUNC(in);                                                     \
        hn::StoreN(x, d, output_array + i, 1);                                 \
      }                                                                        \
    } else if (IS_UNARY_CONT(input_array, output_array)) {                     \
      size_t full = size & -hn::Lanes(d);                                      \
      size_t remainder = size - full;                                          \
      for (size_t i = 0; i < full; i += hn::Lanes(d)) {                        \
        const auto in = hn::LoadU(d, input_array + i);                         \
        auto x = FUNC(in);                                                     \
        hn::StoreU(x, d, output_array + i);                                    \
      }                                                                        \
      if (remainder) {                                                         \
        const auto in = hn::LoadN(d, input_array + full, remainder);           \
        auto x = FUNC(in);                                                     \
        hn::StoreN(x, d, output_array + full, remainder);                      \
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

SUPER(Rint, hn::Round)

HWY_ATTR void DOUBLE_HWRint(char **args, npy_intp const *dimensions, npy_intp const *steps) {
  SuperRint<npy_double>(args, dimensions, steps);
}

HWY_ATTR void FLOAT_HWRint(char **args, npy_intp const *dimensions, npy_intp const *steps) {
  SuperRint<npy_float>(args, dimensions, steps);
}

}
}

#if HWY_ONCE
namespace numpy {

HWY_EXPORT(FLOAT_HWRint);
HWY_EXPORT(DOUBLE_HWRint);

extern "C" {

NPY_NO_EXPORT void
DOUBLE_rint(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  auto dispatcher = HWY_DYNAMIC_POINTER(DOUBLE_HWRint);
  return dispatcher(args, dimensions, steps);
}

NPY_NO_EXPORT void
FLOAT_rint(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
  auto dispatcher = HWY_DYNAMIC_POINTER(FLOAT_HWRint);
  return dispatcher(args, dimensions, steps);
}

} // extern "C"
} // numpy
#endif

