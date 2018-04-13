// #include "aten/src/ATen/SmallVector.h"
#include "llvm/ADT/SmallVector.h"
#include <stdio.h>
#include <stdint.h>

template <typename T>
struct asdd {
 public:
  llvm::SmallVector<int64_t, 8> a__;
  llvm::SmallVector<int64_t, 8> b__;
  llvm::SmallVector<int64_t, 8> c__;

  int64_t* a_;
  int64_t* b_;
  int64_t* c_;
  asdd(asdd const&) = delete;
  void operator=(asdd const& x) = delete;
  asdd(asdd&&) = default;
  asdd() : a__(8, 0), b__(8, 0), c__(8, 0) {
    a_ = a__.data();
    b_ = b__.data();
    c_ = c__.data();
  }
};

template <typename T>
struct asdd1 {
 public:
  int64_t a_[8];
  int64_t b_[8];
  int64_t c_[8];
  asdd1(asdd1 const&) = delete;
  void operator=(asdd1 const& x) = delete;
  asdd1(asdd1&&) = default;
  asdd1() {
    for (size_t i = 0; i < 8; i++) {
      a_[i] = 0;
      b_[i] = 0;
      c_[i] = 0;
    }
  }
};

int main() {
  int64_t agg = 0;
  for (size_t all_count = 0; all_count < 1000; all_count++) {
    auto bb = asdd<float>();
//    auto bb = asdd1<float>();
    for (size_t t = 0; t < 8; t++) {
      bb.a_[t] = t + 1 + all_count;
      bb.b_[t] = t + 2 + all_count;
      bb.c_[t] = t + 3 + all_count;
    }
    agg += bb.a_[1] + bb.b_[1] + bb.c_[1];
    for (size_t count = 0; count < 10000000; count++) {
      for (size_t t = 0; t < 8; t++) {
        bb.a_[t] = bb.a_[t] + count + t;
        bb.b_[t] = bb.b_[t] + count + t;
        bb.c_[t] = bb.c_[t] + count + t;
      }
    agg += bb.a_[1] + bb.b_[1] + bb.c_[1];
    }
  }
  printf("agg2: %ld\n", agg);
}
