#pragma once

#include <ATen/Half.h>
#include <ATen/Type.h>
#include <stdint.h>

// Defines the accumulation type for a scalar type.
// Example:
//   using accscalar_t = cpu::acc_type<scalar_t>;

namespace at { namespace cpu {

template <typename T>
struct AccumulateType { };

template <> struct AccumulateType<float> { using type = float; };
template <> struct AccumulateType<double> { using type = double; };
template <> struct AccumulateType<int8_t> { using type = int64_t; };
template <> struct AccumulateType<uint8_t> { using type = int64_t; };
template <> struct AccumulateType<char> { using type = int64_t; };
template <> struct AccumulateType<int16_t> { using type = int64_t; };
template <> struct AccumulateType<int32_t> { using type = int64_t; };
template <> struct AccumulateType<int64_t> { using type = int64_t; };


template<typename T>
using acc_type = typename AccumulateType<T>::type;

template <typename T>
struct ATenAccumulateType { };

template <> struct ATenAccumulateType<float>   { constexpr static at::ScalarType type = at::ScalarType::Float; }; 
template <> struct ATenAccumulateType<double>  { constexpr static at::ScalarType type = at::ScalarType::Double; }; 
template <> struct ATenAccumulateType<int8_t>  { constexpr static at::ScalarType type = at::ScalarType::Long; }; 
template <> struct ATenAccumulateType<uint8_t> { constexpr static at::ScalarType type = at::ScalarType::Long; }; 
template <> struct ATenAccumulateType<int32_t> { constexpr static at::ScalarType type = at::ScalarType::Long; }; 
template <> struct ATenAccumulateType<int64_t> { constexpr static at::ScalarType type = at::ScalarType::Long; }; 
template <> struct ATenAccumulateType<int16_t> { constexpr static at::ScalarType type = at::ScalarType::Long; }; 


}
}
