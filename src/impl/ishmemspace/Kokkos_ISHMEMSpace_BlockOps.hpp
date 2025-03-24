//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Contact: Jan Ciesko (jciesko@sandia.gov)
//
//@HEADER

#ifndef KOKKOS_REMOTESPACES_ISHMEM_BLOCK_OPS_HPP
#define KOKKOS_REMOTESPACES_ISHMEM_BLOCK_OPS_HPP

#include <ishmem.h>
#include <type_traits>

namespace Kokkos {
namespace Impl {

#define KOKKOS_REMOTESPACES_PUT(type, op)                               \
  static void shmem_block_type_put(type *dst, const type *src,          \
                                              size_t nelems, int pe) {  \
    op(dst, src, nelems, pe);                                           \
  }

#define KOKKOS_REMOTESPACES_GET(type, op)                               \
  static void shmem_block_type_get(type *dst, const type *src,          \
                                              size_t nelems, int pe) {  \
    op(dst, src, nelems, pe);                                           \
  }

#ifdef KRS_USES_NBI_BLOCK
#error Not supported
#elif defined(KRS_USES_NBI)
KOKKOS_REMOTESPACES_PUT(char, ishmem_char_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned char, ishmem_uchar_put_nbi)
KOKKOS_REMOTESPACES_PUT(short, ishmem_short_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned short, ishmem_ushort_put_nbi)
KOKKOS_REMOTESPACES_PUT(int, ishmem_int_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned int, ishmem_uint_put_nbi)
KOKKOS_REMOTESPACES_PUT(long, ishmem_long_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned long, ishmem_ulong_put_nbi)
KOKKOS_REMOTESPACES_PUT(long long, ishmem_longlong_put_nbi)
KOKKOS_REMOTESPACES_PUT(unsigned long long, ishmem_ulonglong_put_nbi)
KOKKOS_REMOTESPACES_PUT(float, ishmem_float_put_nbi)
KOKKOS_REMOTESPACES_PUT(double, ishmem_double_put_nbi)
KOKKOS_REMOTESPACES_GET(char, ishmem_char_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned char, ishmem_uchar_get_nbi)
KOKKOS_REMOTESPACES_GET(short, ishmem_short_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned short, ishmem_ushort_get_nbi)
KOKKOS_REMOTESPACES_GET(int, ishmem_int_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned int, ishmem_uint_get_nbi)
KOKKOS_REMOTESPACES_GET(long, ishmem_long_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned long, ishmem_ulong_get_nbi)
KOKKOS_REMOTESPACES_GET(long long, ishmem_longlong_get_nbi)
KOKKOS_REMOTESPACES_GET(unsigned long long, ishmem_ulonglong_get_nbi)
KOKKOS_REMOTESPACES_GET(float, ishmem_float_get_nbi)
KOKKOS_REMOTESPACES_GET(double, ishmem_double_get_nbi)
#else
KOKKOS_REMOTESPACES_PUT(char, ishmem_char_put)
KOKKOS_REMOTESPACES_PUT(unsigned char, ishmem_uchar_put)
KOKKOS_REMOTESPACES_PUT(short, ishmem_short_put)
KOKKOS_REMOTESPACES_PUT(unsigned short, ishmem_ushort_put)
KOKKOS_REMOTESPACES_PUT(int, ishmem_int_put)
KOKKOS_REMOTESPACES_PUT(unsigned int, ishmem_uint_put)
KOKKOS_REMOTESPACES_PUT(long, ishmem_long_put)
KOKKOS_REMOTESPACES_PUT(unsigned long, ishmem_ulong_put)
KOKKOS_REMOTESPACES_PUT(long long, ishmem_longlong_put)
KOKKOS_REMOTESPACES_PUT(unsigned long long, ishmem_ulonglong_put)
KOKKOS_REMOTESPACES_PUT(float, ishmem_float_put)
KOKKOS_REMOTESPACES_PUT(double, ishmem_double_put)
KOKKOS_REMOTESPACES_GET(char, ishmem_char_get)
KOKKOS_REMOTESPACES_GET(unsigned char, ishmem_uchar_get)
KOKKOS_REMOTESPACES_GET(short, ishmem_short_get)
KOKKOS_REMOTESPACES_GET(unsigned short, ishmem_ushort_get)
KOKKOS_REMOTESPACES_GET(int, ishmem_int_get)
KOKKOS_REMOTESPACES_GET(unsigned int, ishmem_uint_get)
KOKKOS_REMOTESPACES_GET(long, ishmem_long_get)
KOKKOS_REMOTESPACES_GET(unsigned long, ishmem_ulong_get)
KOKKOS_REMOTESPACES_GET(long long, ishmem_longlong_get)
KOKKOS_REMOTESPACES_GET(unsigned long long, ishmem_ulonglong_get)
KOKKOS_REMOTESPACES_GET(float, ishmem_float_get)
KOKKOS_REMOTESPACES_GET(double, ishmem_double_get)
#endif

#undef KOKKOS_REMOTESPACES_PUT
#undef KOKKOS_REMOTESPACES_GET

template <class T, class Traits, typename Enable = void>
struct ISHMEMBlockDataElement {};

template <class T, class Traits>
struct ISHMEMBlockDataElement<T, Traits> {
  typedef const T const_value_type;
  typedef T non_const_value_type;
  T *src;
  T *dst;
  size_t nelems;
  int pe;

  KOKKOS_INLINE_FUNCTION
  ISHMEMBlockDataElement(T *src_, T *dst_, size_t size_, int pe_)
      : src(src_), dst(dst_), nelems(size_), pe(pe_) {}

  KOKKOS_INLINE_FUNCTION
  void put() const { shmem_block_type_put(dst, src, nelems, pe); }

  KOKKOS_INLINE_FUNCTION
  void get() const { shmem_block_type_get(dst, src, nelems, pe); }
};

}  // namespace Impl
}  // namespace Kokkos

#endif  // KOKKOS_REMOTESPACES_ISHMEM_BLOCK_OPS_HPP
