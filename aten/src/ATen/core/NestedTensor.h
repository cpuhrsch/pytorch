#pragma once

#include <c10/core/Device.h>
#include <c10/core/Layout.h>
#include <c10/core/MemoryFormat.h>
#include <c10/core/QScheme.h>
#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>
#include <c10/core/Storage.h>
#include <ATen/core/TensorAccessor.h>
#include <c10/core/TensorImpl.h>
#include <c10/core/UndefinedTensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Optional.h>
#include <c10/util/intrusive_ptr.h>
#include <ATen/core/DeprecatedTypePropertiesRegistry.h>
#include <ATen/core/DeprecatedTypeProperties.h>
#include <ATen/core/EnableNamedTensor.h>
#include <ATen/core/NamedTensor.h>

namespace at {

class NestedTensor {
public:
  NestedTensor(){};
  bool defined() const {
    return true;
  }
};

} // namespace at
