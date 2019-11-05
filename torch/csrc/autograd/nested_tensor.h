#pragma once

#include <torch/csrc/utils/python_stub.h>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/edge.h>
#include <torch/csrc/autograd/function_hook.h>
#include <torch/csrc/autograd/cpp_hook.h>

#include <ATen/ATen.h>
#include <ATen/NamedTensorUtils.h>
#include <c10/util/Exception.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace torch { namespace autograd {

struct Node;

struct TORCH_API NestedTensor {
  /// Default constructor.
  NestedTensor() = default;

  // Gradient Node and Edges
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  const std::shared_ptr<Node>& grad_fn() const;

  Node* grad_fn_unsafe() const;

private:
  /// Set the gradient accumulator of the `Variable`. This is only applicable to
  /// leaf variables. Interior variables should call `set_gradient_edge()`.
  void set_grad_accumulator(std::weak_ptr<Node> grad_accumulator);

  // Only user of set_grad_accumulator
  friend class SavedVariable;

public:
  /// Attempts to get a pointer to the gradient accumulator of the `Variable`,
  /// if it still exists. If the gradient accumulator function has been
  /// destroyed, returns a `nullptr`.
  std::shared_ptr<Node> try_get_grad_accumulator() const;

  /// Gets the gradient accumulator of the `Variable` if it has one, or else
  /// create one on the fly and return it.
  std::shared_ptr<Node> grad_accumulator() const;

  /// Returns the "canonical" gradient edge of this `Variable`, i.e. either the
  /// gradient function if this is an interior `Variable`, or the gradient
  /// accumulator otherwise. If the `Variable` is interior, the returned `Edge`
  /// will store the input index of the `Node` to which this variable is
  /// connected in its `input_nr` field. For leaves, the `input_nr` is always
  /// zero. Note that `set_gradient_edge` and `gradient_edge` are not
  /// symmetric. You must use `set_gradient_edge` to set the `grad_fn` and
  /// `set_grad_accumulator` to set the accumulator.
  Edge gradient_edge() const {
    // If grad_fn is null (as is the case for a leaf node), we instead
    // interpret the gradient function to be a gradient accumulator, which will
    // accumulate its inputs into the grad property of the variable. These
    // nodes get suppressed in some situations, see "suppress gradient
    // accumulation" below. Note that only variables which have `requires_grad =
    // True` can have gradient accumulators.
    if (const auto& gradient = grad_fn()) {
      return Edge(gradient, output_nr());
    } else {
      return Edge(grad_accumulator(), 0);
    }
  }

  /// Returns a copy of this `Variable` that is detached from its autograd graph
  /// and has a blank version. This method is OK to call if the `Variable` is a
  /// view.
  /// NOTE: Previously, if we change the tensor metadata (e.g. sizes / strides /
  /// storage / storage_offset) of a tensor created from `detach()`, those metadata
  /// in the original tensor will also be updated. However, the new behavior is that
  /// those metadata changes to the detached tensor will not update the original tensor
  /// anymore, and in the `detach()` function we need to set `allow_tensor_metadata_change_`
  /// to false to make such changes explicitly illegal, in order to prevent users from
  /// changing metadata of the detached tensor and expecting the original tensor to also
  /// be updated.
  NestedTensor detach() const;

  /// Like `detach()`, but removes this `Variable` in-place. This method may
  /// only be called on non-view `Variable`s. You can use `is_view()` to check
  /// this. If this `Variable` is a view, throws an `std::runtime_error()`.
  void detach_();

  /// Computes the gradient of current tensor w.r.t. graph leaves.
  void backward(
      const NestedTensor& gradient,
      bool keep_graph,
      bool create_graph) const;

  /// Sets the tensor data held by this `Variable` to be the same as `new_data`.
  /// It requires that `new_data` and `Variable` have compatible tensor type, by
  /// checking `_has_compatible_shallow_copy_type(this, new_data)`.
  void set_data(const at::NestedTensor &new_data) const;

  /// Set the gradient edge -- i.e. `grad_fn` and `input_nr` -- of the
  /// `Variable`.
  /// NOTE: This will always set the `grad_fn`, even if this is a leaf variable,
  /// and never the `grad_accumulator`. For the latter, use
  /// `set_grad_accumulator`. This allows late construction of an interior
  /// `Variable`.
  void set_gradient_edge(Edge edge) noexcept;

  /// Returns the input index of the gradient `Node` to which this
  /// `Variable` is connected.  Note: input indexes of the gradient `Node`
  /// correspond to output indexes of the corresponding forward `Node`.
  uint32_t output_nr() const noexcept;

  /// True if this `Variable` is a leaf and thus does not have a `grad_fn`.
  bool is_leaf() const noexcept;

  // Versions
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Increments the version count of this `Variable`.
  void bump_version() noexcept;
  void set_version_counter(const c10::VariableVersion& version_counter) noexcept;

  /// Retrieves this `Variable`s version counter.
  const c10::VariableVersion& version_counter() const noexcept;

  /// Retrieves the current value of the `Variable`'s version counter.
  /// Equivalent to calling `version_counter().current_version()`.
  uint32_t current_version() const noexcept;

  // Autograd Graph Interaction
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Update the `grad_fn` of an existing Variable. Called after in-place
  /// modifications.
  ///
  /// For View Variables:
  /// Called after in-place modifications. Modifies the grad_fn of the base
  /// Variable.
  void rebase_history(Edge gradient_edge);

  // Hooks
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  void add_hook(std::shared_ptr<FunctionPreHook> hook);
  const std::vector<std::shared_ptr<FunctionPreHook>>& hooks() const noexcept;
  void clear_hooks();

  template <typename T>
  using hook_return_void_t = c10::guts::enable_if_t<std::is_void<typename std::result_of<T&(NestedTensor)>::type>::value, unsigned>;
  template <typename T>
  using hook_return_var_t = c10::guts::enable_if_t<std::is_same<typename std::result_of<T&(NestedTensor)>::type, NestedTensor>::value, unsigned>;
  // Remove hook at given position
  void remove_hook(unsigned pos);

  // Returns the index of the hook in the list which can be used to remove hook
  // Register a hook with no return value
  template <typename T>
  hook_return_void_t<T> register_hook(T&& hook);
  // Register a hook with variable return value
  template <typename T>
  hook_return_var_t<T> register_hook(T&& hook);

  // View Variables
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  /// Returns true if this `Variable` is a view of another `Variable`.
  bool is_view() const noexcept;

  /// Returns the `Variable` that this `Variable` is a view of. If this
  /// `Variable` is not a view, throw a `std::runtime_error`.
  const NestedTensor& base() const;

  // Miscellaneous
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  void set_name(const std::string& name);
  const std::string& name() const noexcept;

  PyObject* pyobj() const noexcept;
  void set_pyobj(PyObject* pyobj) noexcept;

 private:
  struct AutogradMeta;

 public:
  NestedTensor::AutogradMeta* get_autograd_meta() const noexcept;

 private:
  struct DifferentiableViewMeta;

  // Private Methods
  //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

  void create_cpp_hook();
};

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//                            Variable::AutogradMeta
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Each `Variable` has one unique `AutogradMeta` struct, which stores autograd
/// metadata fields that are necessary for tracking the Variable's autograd history.

struct TORCH_API NestedTensor::AutogradMeta : public c10::AutogradMetaInterface {
  std::string name_;

  NestedTensor grad_;
  std::shared_ptr<Node> grad_fn_;
  std::weak_ptr<Node> grad_accumulator_;

  std::vector<std::shared_ptr<FunctionPreHook>> hooks_;
  std::shared_ptr<hooks_list> cpp_hooks_list;

  // Only meaningful on leaf variables (must be false otherwise)
  bool requires_grad_;

  bool is_view_;

  // The "output number" of this variable; e.g., if this variable
  // was the second output of a function, then output_nr == 1.
  // We use this to make sure we can setup the backwards trace
  // correctly when this variable is passed to another function.
  uint32_t output_nr_;

  // Mutex to ensure that concurrent read operations that modify internal
  // state are still thread-safe. Used by grad_fn() and
  // grad_accumulator().
  std::mutex mutex_;

  /// Sets the `requires_grad` property of `Variable`. This should be true for
  /// leaf variables that want to accumulate gradients, and false for all other
  /// variables.
  void set_requires_grad(bool requires_grad, at::NestedTensor* self_impl) override {
    TORCH_CHECK(
      !requires_grad || at::isFloatingType(at::typeMetaToScalarType(self_impl->dtype())),
      "Only Tensors of floating point dtype can require gradients");
    requires_grad_ = requires_grad;
  }

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_;
  }

  /// Accesses the gradient `Variable` of this `Variable`.
  NestedTensor& grad() override {
    return grad_;
  }

  const NestedTensor& grad() const override {
    return grad_;
  }

  AutogradMeta(
    at::NestedTensor* self_impl,
    bool requires_grad = false,
    Edge gradient_edge = Edge());
};

struct TORCH_API NestedTensor::DifferentiableViewMeta : public NestedTensor::AutogradMeta {
  /// The base `Variable` (never a view).
  NestedTensor base_;

  /// The value of the version_counter at the time grad_fn was created. The
  /// grad_fn field is stale if attr_version !=
  /// version_counter.current_version().
  uint32_t attr_version;

  bool requires_grad() const override {
    return requires_grad_ || grad_fn_ || (is_view_ && base_.requires_grad());
  }

  DifferentiableViewMeta(at::NestedTensor* self_impl, NestedTensor base, Edge gradient_edge);
  ~DifferentiableViewMeta();
};

inline NestedTensor make_NestedTensor_view(
    NestedTensor base,
    at::NestedTensor data,
    bool is_differentiable = true,
    bool allow_tensor_metadata_change = true,
    Edge gradient_edge = Edge()) {
  if (data.defined()) {
    if (is_differentiable) {
      /// Differentiable view. Track history with DifferentiableViewMeta.
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/0,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      data_impl_copy->set_autograd_meta(c10::guts::make_unique<NestedTensor::DifferentiableViewMeta>(
        data_impl_copy.get(), std::move(base), std::move(gradient_edge)));
      return NestedTensor(data_impl_copy);
    } else {
      /// Non-differentiable view. Just share version counter.
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/base.version_counter(),
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      data_impl_copy->set_autograd_meta(c10::guts::make_unique<NestedTensor::AutogradMeta>(
        data_impl_copy.get(), false, std::move(gradient_edge)));
      return NestedTensor(data_impl_copy);
    }
  }
  return NestedTensor();
}

inline NestedTensor make_variable(
    at::NestedTensor data,
    bool requires_grad = false,
    bool allow_tensor_metadata_change = true) {
  TORCH_CHECK(
      !data.is_variable(),
      "Must not create a new variable from a variable, use its .tensor_data()");
  if (data.defined()) {
    if (data.getIntrusivePtr().use_count() == 1 && data.getIntrusivePtr()->unique_version()) {
      auto data_impl = data.getIntrusivePtr();
      data_impl->set_allow_tensor_metadata_change(allow_tensor_metadata_change);
      data_impl->set_autograd_meta(c10::guts::make_unique<NestedTensor::AutogradMeta>(data_impl.get(), requires_grad));
      return NestedTensor(std::move(data_impl));
    } else {
      auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
        /*version_counter=*/0,
        /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
      data_impl_copy->set_autograd_meta(c10::guts::make_unique<NestedTensor::AutogradMeta>(
        data_impl_copy.get(), requires_grad));
      return NestedTensor(data_impl_copy);
    }
  }
  return NestedTensor();
}

inline NestedTensor make_variable(
    at::NestedTensor data,
    Edge gradient_edge,
    bool allow_tensor_metadata_change = true) {
  TORCH_CHECK(
      !data.is_variable(),
      "Must not create a new NestedTensor from a NestedTensor, use its .tensor_data()");
  if (data.defined()) {
    auto data_impl_copy = data.getIntrusivePtr()->shallow_copy_and_detach(
      /*version_counter=*/0,
      /*allow_tensor_metadata_change=*/allow_tensor_metadata_change);
    data_impl_copy->set_autograd_meta(c10::guts::make_unique<NestedTensor::AutogradMeta>(
      data_impl_copy.get(), false, std::move(gradient_edge)));
    return NestedTensor(data_impl_copy);
  }
  return NestedTensor();
}

// Tensor Conversion
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/// Downcasts the `Tensor` reference to a `NestedTensor` reference. If compiling
/// in DEBUG mode and the tensor's dynamic type is not in fact `NestedTensor`,
/// throws a `std::invalid_argument` exception.
inline NestedTensor& as_NestedTensor_ref(at::NestedTensor& tensor) {
  TORCH_CHECK(
      tensor.is_NestedTensor(),
      "Attempted to cast a Tensor to a NestedTensor, but "
      "the dynamic type of the value is not NestedTensor.");
  return static_cast<NestedTensor&>(tensor);
}

inline const NestedTensor& as_NestedTensor_ref(const at::NestedTensor& tensor) {
  TORCH_CHECK(
      tensor.is_NestedTensor(),
      "Attempted to cast a Tensor to a NestedTensor, but "
      "the dynamic type of the value is not NestedTensor.");
  return static_cast<const NestedTensor&>(tensor);
}

inline at::Tensor NestedTensor::tensor_data() const noexcept {
  auto self_impl_copy = get()->shallow_copy_and_detach(
    /*version_counter=*/get()->version_counter(),
    /*allow_tensor_metadata_change=*/get()->allow_tensor_metadata_change());
  return at::Tensor(self_impl_copy);
}

inline at::Tensor NestedTensor::NestedTensor_data() const noexcept {
  auto self_impl_copy = get()->shallow_copy_and_detach(
    /*version_counter=*/0,
    /*allow_tensor_metadata_change=*/false);
  self_impl_copy->set_autograd_meta(c10::guts::make_unique<NestedTensor::AutogradMeta>(self_impl_copy.get(), false));
  return at::Tensor(self_impl_copy);
}

// Gradient Node and Edges
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline Node* NestedTensor::grad_fn_unsafe() const {
  return get_autograd_meta()->grad_fn_.get();
}

inline void NestedTensor::set_grad_accumulator(
    std::weak_ptr<Node> grad_accumulator) {
  get_autograd_meta()->grad_accumulator_ = std::move(grad_accumulator);
}

inline std::shared_ptr<Node> NestedTensor::try_get_grad_accumulator() const {
  return get_autograd_meta()->grad_accumulator_.lock();
}

inline NestedTensor NestedTensor::detach() const {
  auto var = make_NestedTensor_view(*this, *this, /*is_differentiable=*/false, /*allow_tensor_metadata_change=*/false, Edge());
#ifdef BUILD_NAMEDTENSOR
  at::namedinference::propagate_names(var, *this);
#endif
  return var;
}

inline void NestedTensor::set_gradient_edge(Edge edge) noexcept {
  get_autograd_meta()->grad_fn_ = std::move(edge.function);
  get_autograd_meta()->output_nr_ = edge.input_nr;
}

inline uint32_t NestedTensor::output_nr() const noexcept {
  return get_autograd_meta()->output_nr_;
}

inline bool NestedTensor::is_leaf() const noexcept {
  return get_autograd_meta()->grad_fn_ == nullptr;
}

// Versions
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline void NestedTensor::set_version_counter(
    const c10::NestedTensorVersion& version_counter) noexcept {
  unsafeGetTensorImpl()->set_version_counter(version_counter);
}

inline void NestedTensor::bump_version() noexcept {
  unsafeGetTensorImpl()->bump_version();
}

inline uint32_t NestedTensor::current_version() const noexcept {
  return unsafeGetTensorImpl()->version_counter().current_version();
}

inline const c10::NestedTensorVersion& NestedTensor::version_counter() const noexcept {
  return unsafeGetTensorImpl()->version_counter();
}

// Hooks
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline void NestedTensor::add_hook(std::shared_ptr<FunctionPreHook> hook) {
  get_autograd_meta()->hooks_.push_back(std::move(hook));
}

inline const std::vector<std::shared_ptr<FunctionPreHook>>& NestedTensor::hooks()
    const noexcept {
  return get_autograd_meta()->hooks_;
}

inline void NestedTensor::clear_hooks() {
  get_autograd_meta()->hooks_.clear();
}

template <typename T>
auto NestedTensor::register_hook(T&& hook) -> NestedTensor::hook_return_void_t<T> {
  TORCH_CHECK(requires_grad(), "cannot register a hook on a NestedTensor that "
                           "doesn't require gradient");
  auto &list = get_autograd_meta()->cpp_hooks_list;
  if(!list) {
    create_cpp_hook();
  }
  unsigned idx = list->size();
  // Return the grad argument in case of a hook with void return type to have an
  // std::function with NestedTensor return type
  std::function<void(NestedTensor)> fn(hook);
  list->emplace_back([fn](NestedTensor grad){
   fn(grad);
    return NestedTensor();});
  return idx;
}

template <typename T>
auto NestedTensor::register_hook(T&& hook) -> NestedTensor::hook_return_var_t<T> {
  TORCH_CHECK(requires_grad(), "cannot register a hook on a NestedTensor that "
                           "doesn't require gradient");
  auto &list = get_autograd_meta()->cpp_hooks_list;
  if(!list) {
    create_cpp_hook();
  }
  unsigned idx = list->size();
  list->push_back(hook);
  return idx;
}

// View NestedTensors
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline bool NestedTensor::is_view() const noexcept {
  return get_autograd_meta()->is_view_;
}

inline const NestedTensor& NestedTensor::base() const {
  if (is_view()) {
    auto diff_view_meta = static_cast<NestedTensor::DifferentiableViewMeta*>(get_autograd_meta());
    return diff_view_meta->base_;
  } else {
    throw std::runtime_error("Can't get base of non-view NestedTensor");
  }
}

// Miscellaneous
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline void NestedTensor::set_name(const std::string& name) {
  get_autograd_meta()->name_ = name;
}

inline const std::string& NestedTensor::name() const noexcept {
  return get_autograd_meta()->name_;
}

inline void NestedTensor::set_pyobj(PyObject* pyobj) noexcept {
  get()->set_pyobj(pyobj);
}

inline PyObject* NestedTensor::pyobj() const noexcept {
  return get()->pyobj();
}

inline NestedTensor::AutogradMeta* NestedTensor::get_autograd_meta() const noexcept {
  return static_cast<NestedTensor::AutogradMeta*>(get()->autograd_meta());
}

// Private Methods
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

inline NestedTensor::NestedTensor(c10::intrusive_ptr<at::TensorImpl> self)
    : at::Tensor(std::move(self)) {}

inline at::TensorImpl* NestedTensor::get() const {
  TORCH_CHECK(defined(), "Called NestedTensor::get() on an undefined NestedTensor");
  return unsafeGetTensorImpl();
}
}} // namespace torch::autograd
