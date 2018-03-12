#pragma once

#if defined(_MSC_VER)
/* Microsoft C/C++-compatible compiler */
#include <intrin.h>
#elif defined(__GNUC__) && (defined(__x86_64__) || defined(__i386__))
/* GCC-compatible compiler, targeting x86/x86-64 */
#include <x86intrin.h>
#elif defined(__GNUC__) && defined(__ARM_NEON__)
/* GCC-compatible compiler, targeting ARM with NEON */
#include <arm_neon.h>
#elif defined(__GNUC__) && defined(__IWMMXT__)
/* GCC-compatible compiler, targeting ARM with WMMX */
#include <mmintrin.h>
#elif (defined(__GNUC__) || defined(__xlC__)) &&                               \
    (defined(__VEC__) || defined(__ALTIVEC__))
/* XLC or GCC-compatible compiler, targeting PowerPC with VMX/VSX */
#include <altivec.h>
#elif defined(__GNUC__) && defined(__SPE__)
/* GCC-compatible compiler, targeting PowerPC with SPE */
#include <spe.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>


// NOTE:
// If you specialize on a type, you must define all operations!
// C arrays and intrinsic types don't mix

namespace at {
namespace native {
namespace pow2vec {

// W must be a power of 2 and specifies the number of bytes. 
// 32 bytes is 256 bits which is the size of a AVX/AVX2 register
template <size_t W, class T> class Pow2Vec {
public:
  T values[W / sizeof(T)]; // Mimics AVX behavior
  inline void load(const T *ptr) { 
    std::memcpy(values, ptr, W); 
  };
  inline void store(T *ptr) { std::memcpy(ptr, values, W); }
  inline size_t size() { return W / sizeof(T); }
  inline void operator=(const Pow2Vec<W, T> &b) {
    std::memcpy(values, b.values, W);
  }
};

template <class T>
T operator+(const T &a, const T &b) {
  T c = T();
  for (size_t i = 0; i < c.size(); i++) {
    c.values[i] = a.values[i] + b.values[i];
  }
  return c;
}

template <class T>
T operator*(const T &a, const T &b) {
  T c = T();
  for (size_t i = 0; i < c.size(); i++) {
    c.values[i] = a.values[i] * b.values[i];
  }
  return c;
}

template <class AW, class AT>
AW vecsum(const AT &a, const AT &b) {
  AW c = AW();
  for (size_t i = 0; i < c.size(); i++) {
    c.values[i] = a.values[i] + b.values[i];
  }
  return c;
}

template <class AW, class AT>
AW vecmult(const AT &a, const AT &b) {
  AW c = AW();
  for (size_t i = 0; i < c.size(); i++) {
    c.values[i] = a.values[i] * b.values[i];
  }
  return c;
}

#ifdef __AVX__
template <> class Pow2Vec<32, float> {
public:
  __m256 values;
  Pow2Vec<32, float>() {}
  inline void load(const float *ptr) { values = _mm256_loadu_ps(ptr); }
  inline void store(float *ptr) { _mm256_storeu_ps(ptr, values); }
  inline size_t size() { return 32 / sizeof(float); }
  inline void operator=(const Pow2Vec<32, float> &b) { values = b.values; }
};

template <> class Pow2Vec<32, double> {
public:
  __m256d values;
  Pow2Vec<32, double>() {}
  inline void load(const double *ptr) { values = _mm256_loadu_pd(ptr); }
  inline void store(double *ptr) { _mm256_storeu_pd(ptr, values); }
  inline size_t size() { return 32 / sizeof(double); }
  inline void operator=(const Pow2Vec<32, double> &b) { values = b.values; }
};

template <>
Pow2Vec<32, float> inline operator+(const Pow2Vec<32, float> &a, const Pow2Vec<32, float> &b) {
  Pow2Vec<32, float> c = Pow2Vec<32, float>();
  c.values = _mm256_add_ps(a.values, b.values);
  return c;
}

template <>
Pow2Vec<32, float> inline operator*(const Pow2Vec<32, float> &a, const Pow2Vec<32, float> &b) {
  Pow2Vec<32, float> c = Pow2Vec<32, float>();
  c.values = _mm256_mul_ps(a.values, b.values);
  return c;
}

template <>
Pow2Vec<32, double> inline operator+(const Pow2Vec<32, double> &a,
                                const Pow2Vec<32, double> &b) {
  Pow2Vec<32, double> c = Pow2Vec<32, double>();
  c.values = _mm256_add_pd(a.values, b.values);
  return c;
}

template <>
Pow2Vec<32, double> inline operator*(const Pow2Vec<32, double> &a,
                                const Pow2Vec<32, double> &b) {
  Pow2Vec<32, double> c = Pow2Vec<32, double>();
  c.values = _mm256_mul_pd(a.values, b.values);
  return c;
}
#endif

#ifdef __AVX2__
template <> class Pow2Vec<32, int64_t> {
public:
  __m256i values;
  Pow2Vec<32, int64_t>() {}
  inline void load(const int64_t *ptr) {
    values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
  }
  inline void store(int64_t *ptr) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), values);
  }
  inline size_t size() { return 32 / sizeof(int64_t); }
  inline void operator=(const Pow2Vec<32, int64_t> &b) { values = b.values; }
};

template <> class Pow2Vec<32, int32_t> {
public:
  __m256i values;
  Pow2Vec<32, int32_t>() {}
  inline void load(const int32_t *ptr) {
    values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
  }
  inline void store(int32_t *ptr) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), values);
  }
  inline size_t size() { return 32 / sizeof(int32_t); }
  inline void operator=(const Pow2Vec<32, int32_t> &b) { values = b.values; }
};

template <> class Pow2Vec<32, int16_t> {
public:
  __m256i values;
  Pow2Vec<32, int16_t>() {}
  inline void load(const int16_t *ptr) {
    values = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(ptr));
  }
  inline void store(int16_t *ptr) {
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(ptr), values);
  }
  inline size_t size() { return 32 / sizeof(int16_t); }
  inline void operator=(const Pow2Vec<32, int16_t> &b) { values = b.values; }
};

template <>
Pow2Vec<32, int64_t> inline operator+(const Pow2Vec<32, int64_t> &a,
                                 const Pow2Vec<32, int64_t> &b) {
  Pow2Vec<32, int64_t> c = Pow2Vec<32, int64_t>();
  c.values = _mm256_add_epi64(a.values, b.values);
  return c;
}

template <>
Pow2Vec<32, int32_t> inline operator+(const Pow2Vec<32, int32_t> &a,
                                 const Pow2Vec<32, int32_t> &b) {
  Pow2Vec<32, int32_t> c = Pow2Vec<32, int32_t>();
  c.values = _mm256_add_epi32(a.values, b.values);
  return c;
}

template <>
Pow2Vec<32, int16_t> inline operator+(const Pow2Vec<32, int16_t> &a,
                                 const Pow2Vec<32, int16_t> &b) {
  Pow2Vec<32, int16_t> c = Pow2Vec<32, int16_t>();
  c.values = _mm256_add_epi16(a.values, b.values);
  return c;
}

// AVX2 has no intrinsic for int64_t multiply so it needs to be emulated
// This could be implemented more efficiently using epi32 instructions
// This is also technically avx compatible, but then we'll need AVX
// code for add as well.
template <>
Pow2Vec<32, int64_t> inline operator*(const Pow2Vec<32, int64_t> &a,
                                 const Pow2Vec<32, int64_t> &b) {
  Pow2Vec<32, int64_t> c = Pow2Vec<32, int64_t>();

  int64_t a0 = _mm256_extract_epi64(a.values, 0);
  int64_t a1 = _mm256_extract_epi64(a.values, 1);
  int64_t a2 = _mm256_extract_epi64(a.values, 2);
  int64_t a3 = _mm256_extract_epi64(a.values, 3);

  int64_t b0 = _mm256_extract_epi64(b.values, 0);
  int64_t b1 = _mm256_extract_epi64(b.values, 1);
  int64_t b2 = _mm256_extract_epi64(b.values, 2);
  int64_t b3 = _mm256_extract_epi64(b.values, 3);

  int64_t c0 = a0 * b0;
  int64_t c1 = a1 * b1;
  int64_t c2 = a2 * b2;
  int64_t c3 = a3 * b3;

  c.values = _mm256_set_epi64x(c3, c2, c1, c0);
  return c;
}

template <>
Pow2Vec<32, int32_t> inline operator*(const Pow2Vec<32, int32_t> &a,
                                 const Pow2Vec<32, int32_t> &b) {
  Pow2Vec<32, int32_t> c = Pow2Vec<32, int32_t>();
  c.values = _mm256_mullo_epi32(a.values, b.values);
  return c;
}

template <>
Pow2Vec<32, int16_t> inline operator*(const Pow2Vec<32, int16_t> &a,
                                 const Pow2Vec<32, int16_t> &b) {
  Pow2Vec<32, int16_t> c = Pow2Vec<32, int16_t>();
  c.values = _mm256_mullo_epi16(a.values, b.values);
  return c;
}
#endif
}
}
}
