#define _UMATHMODULE
#define _MULTIARRAYMODULE
#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <type_traits>
#include "numpy/ndarraytypes.h"
#include "numpy/npy_common.h"
#include "numpy/npy_math.h"
#include "numpy/utils.h"
#include "fast_loop_macros.h"
#include "loops_utils.h"
#include <hwy/highway.h>
#include <hwy/aligned_allocator.h>
#include <hwy/print-inl.h>

namespace hn = hwy::HWY_NAMESPACE;

inline float Chypot(float x, float y){
    return npy_hypotf(x, y);
} 

inline double Chypot(double x, double y){
    return npy_hypot(x, y);
}

template <typename T>
HWY_ATTR void SuperCabsolute(char** args,
                             npy_intp const* dimensions,
                             npy_intp const* steps) {
  npy_intp len = dimensions[0];
  npy_intp ssrc = steps[0] / sizeof(T);
  npy_intp sdst = steps[1] / sizeof(T);
  if (!is_mem_overlap(args[0], steps[0], args[1], steps[1], len) &&
      steps[0] % sizeof(T) == 0 && steps[1] % sizeof(T) == 0) {
    const T* src = (T*)args[0];
    T* dst = (T*)args[1];
    const hn::ScalableTag<T> d;
    const int vstep = hn::Lanes(d);
    const int wstep = vstep * 2;
    // const int hstep = vstep / 2;

    using TI = hwy::MakeSigned<T>;
    const hn::Rebind<TI, hn::ScalableTag<T>> di;
    auto indices = hwy::AllocateAligned<TI>(vstep);
    auto load_index = hn::Mul(hn::Iota(di, 0), hn::Set(di, ssrc));
    auto store_index = hn::Mul(hn::Iota(di, 0), hn::Set(di, sdst));

    using vec_f = hn::Vec<decltype(d)>;

    // TODO: Optimize HWYComplexabsolute this by
    // NPY_FINLINE npyv_@sfx@
    // simd_cabsolute_@sfx@(npyv_@sfx@ re, npyv_@sfx@ im)
    // {
    //     const npyv_@sfx@ inf = npyv_setall_@sfx@(@INF@);
    //     const npyv_@sfx@ nan = npyv_setall_@sfx@(@NAN@);

    //     re = npyv_abs_@sfx@(re);
    //     im = npyv_abs_@sfx@(im);
    //     /*
    //      * If real or imag = INF, then convert it to inf + j*inf
    //      * Handles: inf + j*nan, nan + j*inf
    //      */
    //     npyv_@bsfx@ re_infmask = npyv_cmpeq_@sfx@(re, inf);
    //     npyv_@bsfx@ im_infmask = npyv_cmpeq_@sfx@(im, inf);
    //     im = npyv_select_@sfx@(re_infmask, inf, im);
    //     re = npyv_select_@sfx@(im_infmask, inf, re);
    //     /*
    //      * If real or imag = NAN, then convert it to nan + j*nan
    //      * Handles: x + j*nan, nan + j*x
    //      */
    //     npyv_@bsfx@ re_nnanmask = npyv_notnan_@sfx@(re);
    //     npyv_@bsfx@ im_nnanmask = npyv_notnan_@sfx@(im);
    //     im = npyv_select_@sfx@(re_nnanmask, im, nan);
    //     re = npyv_select_@sfx@(im_nnanmask, re, nan);

    //     npyv_@sfx@ larger  = npyv_max_@sfx@(re, im);
    //     npyv_@sfx@ smaller = npyv_min_@sfx@(im, re);
    //     /*
    //      * Calculate div_mask to prevent 0./0. and inf/inf operations in div
    //      */
    //     npyv_@bsfx@ zeromask = npyv_cmpeq_@sfx@(larger, npyv_zero_@sfx@());
    //     npyv_@bsfx@ infmask = npyv_cmpeq_@sfx@(smaller, inf);
    //     npyv_@bsfx@ div_mask = npyv_not_@bsfx@(npyv_or_@bsfx@(zeromask, infmask));

    //     npyv_@sfx@ ratio = npyv_ifdivz_@sfx@(div_mask, smaller, larger);
    //     npyv_@sfx@ hypot = npyv_sqrt_@sfx@(
    //         npyv_muladd_@sfx@(ratio, ratio, npyv_setall_@sfx@(1.0@c@)
    //     ));
    //     return npyv_mul_@sfx@(hypot, larger);
    // }
    auto HWYComplexabsolute = [d](vec_f re, vec_f im) {
      const auto inf = hn::Set(d, std::is_floating_point<T>::value ? NPY_INFINITYF : NPY_INFINITY);
      const auto nan = hn::Set(d, std::is_floating_point<T>::value ? NPY_NANF : NPY_NAN);
      re = hn::Abs(re);
      im = hn::Abs(im);

      /*
       * If real or imag = INF, then convert it to inf + j*inf
       * Handles: inf + j*nan, nan + j*inf
       */
      auto re_infmask = hn::IsInf(re);
      auto im_infmask = hn::IsInf(im);
      im = hn::IfThenElse(re_infmask, inf, im);
      re = hn::IfThenElse(im_infmask, inf, re);
      /*
       * If real or imag = NAN, then convert it to nan + j*nan
       * Handles: x + j*nan, nan + j*x
       */
      auto re_nanmask = hn::IsNaN(re);
      auto im_nanmask = hn::IsNaN(im);
      im = hn::IfThenElse(re_nanmask, nan, im);
      re = hn::IfThenElse(im_nanmask, nan, re);

      auto larger = hn::Max(re, im);
      auto smaller = hn::Min(re, im);
      /*
       * Calculate div_mask to prevent 0./0. and inf/inf operations in div
       */
      auto zeromask = hn::Eq(larger, hn::Zero(d));
      auto infmask = hn::IsInf(larger);
      auto divmask = hn::Not(hn::Or(infmask, zeromask));
      auto one = hn::Set(d, 1);
      auto div = hn::Div(smaller, hn::IfThenElse(divmask, larger, one));
      auto ratio = hn::IfThenElseZero(divmask, div);

      auto hypot = hn::Sqrt(hn::MulAdd(ratio, ratio, hn::Set(d, 1.0)));
      auto result = hn::Mul(hypot, larger);
      return result;
    };

    if (ssrc == 2 && sdst == 1) {
      for (; len >= vstep; len -= vstep, src += wstep, dst += vstep) {
        vec_f re, im;
        hn::LoadInterleaved2(d, src, re, im);
        auto r = HWYComplexabsolute(re, im);
        hn::StoreU(r, d, dst);
      }
    } else {
      for (; len >= vstep;
           len -= vstep, src += ssrc * vstep, dst += sdst * vstep) {
        auto re = hn::GatherIndex(d, src, load_index);
        auto im = hn::GatherIndex(d, src + 1, load_index);
        auto r = HWYComplexabsolute(re, im);
        hn::ScatterIndex(r, d, dst, store_index);
      }
    }
    for (; len > 0; len -= vstep, src += ssrc * vstep, dst += sdst * vstep) {
      auto re = hn::GatherIndexN(d, src, load_index, len);
      auto im = hn::GatherIndexN(d, src + 1, load_index, len);
      auto r = HWYComplexabsolute(re, im);
      hn::ScatterIndexN(r, d, dst, store_index, len);
    }
  } else {
    UNARY_LOOP {
      const T re = ((T*)ip1)[0];
      const T im = ((T*)ip1)[1];
      *((T*)op1) = Chypot(re, im);
    }
  }
}

extern "C" {
NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(CFLOAT_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    SuperCabsolute<float>(args, dimensions, steps);
}

NPY_NO_EXPORT void NPY_CPU_DISPATCH_CURFX(CDOUBLE_absolute)
(char **args, npy_intp const *dimensions, npy_intp const *steps, void *NPY_UNUSED(func))
{
    SuperCabsolute<double>(args, dimensions, steps);
}
}