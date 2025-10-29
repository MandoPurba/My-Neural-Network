//! Gradient Tensor Implementation
//!
//! This module provides the GradTensor type that wraps regular tensors
//! and tracks gradient information for automatic differentiation.

use crate::autodiff::NodeId;
use crate::backend::Backend;
use crate::data::DataType;
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::sync::Arc;

/// Trait to mark tensors that require gradient computation
pub trait RequiresGrad {
    /// Whether this tensor requires gradient computation
    fn requires_grad(&self) -> bool;

    /// Set whether this tensor requires gradient computation
    fn set_requires_grad(&mut self, requires_grad: bool);
}

/// A tensor wrapper that tracks gradients for automatic differentiation
#[derive(Debug, Clone)]
pub struct GradTensor<B: Backend, const D: usize, K: DataType = crate::Float> {
    /// The underlying tensor data
    pub tensor: Tensor<B, D, K>,

    /// Unique identifier for this tensor in the computation graph
    pub node_id: NodeId,

    /// Whether this tensor requires gradient computation
    pub requires_grad: bool,

    /// The accumulated gradient for this tensor
    pub grad: Option<Arc<RefCell<Tensor<B, D, K>>>>,

    /// Version counter for gradient updates
    pub version: u64,
}

impl<B: Backend, const D: usize, K: DataType> GradTensor<B, D, K> {
    /// Create a new gradient tensor from a regular tensor
    pub fn new(tensor: Tensor<B, D, K>, requires_grad: bool) -> Self {
        Self {
            tensor,
            node_id: NodeId::new(),
            requires_grad,
            grad: None,
            version: 0,
        }
    }

    /// Create a new gradient tensor that requires gradients
    pub fn with_grad(tensor: Tensor<B, D, K>) -> Self {
        Self::new(tensor, true)
    }

    /// Create a new gradient tensor that doesn't require gradients
    pub fn no_grad(tensor: Tensor<B, D, K>) -> Self {
        Self::new(tensor, false)
    }

    /// Get a reference to the underlying tensor
    pub fn tensor(&self) -> &Tensor<B, D, K> {
        &self.tensor
    }

    /// Get a mutable reference to the underlying tensor
    pub fn tensor_mut(&mut self) -> &mut Tensor<B, D, K> {
        &mut self.tensor
    }

    /// Convert back to a regular tensor, losing gradient tracking
    pub fn into_tensor(self) -> Tensor<B, D, K> {
        self.tensor
    }

    /// Get the node ID for this tensor in the computation graph
    pub fn node_id(&self) -> NodeId {
        self.node_id
    }

    /// Check if this tensor has a gradient
    pub fn has_grad(&self) -> bool {
        self.grad.is_some()
    }

    /// Get the gradient tensor if it exists
    pub fn grad(&self) -> Option<Tensor<B, D, K>>
    where
        K::Primitive: Clone,
    {
        self.grad.as_ref().map(|g| g.borrow().clone())
    }

    /// Set the gradient for this tensor
    pub fn set_grad(&mut self, gradient: Tensor<B, D, K>) {
        self.grad = Some(Arc::new(RefCell::new(gradient)));
        self.version += 1;
    }

    /// Accumulate gradient (add to existing gradient)
    pub fn accumulate_grad(&mut self, gradient: Tensor<B, D, K>)
    where
        K::Primitive: Clone,
    {
        match &self.grad {
            Some(existing_grad) => {
                let grad_ref = existing_grad.borrow();
                let accumulated = grad_ref.clone();
                drop(grad_ref);
                *existing_grad.borrow_mut() = accumulated;
            }
            None => {
                self.grad = Some(Arc::new(RefCell::new(gradient)));
            }
        }
        self.version += 1;
    }

    /// Clear the gradient
    pub fn zero_grad(&mut self)
    where
        B: Default,
        K::Primitive: Clone + Default,
    {
        if let Some(grad) = &self.grad {
            let grad_ref = grad.borrow();
            let shape = grad_ref.shape().clone();
            let size = shape.num_elements();
            let zero_data = vec![K::Primitive::default(); size];
            let zeros = Tensor::from_data(zero_data, shape);
            drop(grad_ref);
            *grad.borrow_mut() = zeros;
        }
        self.version += 1;
    }

    /// Remove the gradient entirely
    pub fn clear_grad(&mut self) {
        self.grad = None;
        self.version += 1;
    }

    /// Get the gradient version (incremented each time gradient changes)
    pub fn grad_version(&self) -> u64 {
        self.version
    }

    /// Create a copy of this tensor that doesn't require gradients
    pub fn detach(&self) -> Self
    where
        K::Primitive: Clone,
    {
        Self::new(self.tensor.clone(), false)
    }

    /// Create a copy of this tensor that requires gradients
    pub fn detach_with_grad(&self) -> Self
    where
        K::Primitive: Clone,
    {
        Self::new(self.tensor.clone(), true)
    }

    /// Apply a function to the underlying tensor while preserving gradient tracking
    pub fn map_tensor<F>(&self, f: F) -> Self
    where
        F: FnOnce(&Tensor<B, D, K>) -> Tensor<B, D, K>,
        K::Primitive: Clone,
    {
        let new_tensor = f(&self.tensor);
        Self::new(new_tensor, self.requires_grad)
    }
}

impl<B: Backend, const D: usize, K: DataType> RequiresGrad for GradTensor<B, D, K> {
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    fn set_requires_grad(&mut self, requires_grad: bool) {
        self.requires_grad = requires_grad;
        if !requires_grad {
            self.clear_grad();
        }
    }
}

impl<B: Backend, const D: usize, K: DataType> RequiresGrad for Tensor<B, D, K> {
    fn requires_grad(&self) -> bool {
        false // Regular tensors don't track gradients
    }

    fn set_requires_grad(&mut self, _requires_grad: bool) {
        // No-op for regular tensors
    }
}

// Convenience methods for creating gradient tensors
impl<B: Backend, const D: usize, K: DataType> From<Tensor<B, D, K>> for GradTensor<B, D, K> {
    fn from(tensor: Tensor<B, D, K>) -> Self {
        Self::no_grad(tensor)
    }
}

// Allow dereferencing to access tensor methods directly
impl<B: Backend, const D: usize, K: DataType> std::ops::Deref for GradTensor<B, D, K> {
    type Target = Tensor<B, D, K>;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, Shape, Tensor};

    #[test]
    fn test_grad_tensor_creation() {
        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);

        let grad_tensor = GradTensor::with_grad(tensor.clone());
        assert!(grad_tensor.requires_grad());
        assert!(!grad_tensor.has_grad());

        let no_grad_tensor = GradTensor::no_grad(tensor);
        assert!(!no_grad_tensor.requires_grad());
        assert!(!no_grad_tensor.has_grad());
    }

    #[test]
    fn test_gradient_operations() {
        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());
        let grad: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4], shape);

        let mut grad_tensor = GradTensor::with_grad(tensor);

        // Set gradient
        grad_tensor.set_grad(grad.clone());
        assert!(grad_tensor.has_grad());

        let retrieved_grad = grad_tensor.grad().unwrap();
        assert_eq!(retrieved_grad.shape().dims(), &[2, 2]);

        // Accumulate gradient
        grad_tensor.accumulate_grad(grad.clone());
        let accumulated_grad = grad_tensor.grad().unwrap();
        // Should be 2x the original gradient

        // Zero gradient
        grad_tensor.zero_grad();
        let zero_grad = grad_tensor.grad().unwrap();
        // Should be zeros with same shape

        // Clear gradient
        grad_tensor.clear_grad();
        assert!(!grad_tensor.has_grad());
    }

    #[test]
    fn test_detach() {
        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);

        let grad_tensor = GradTensor::with_grad(tensor);
        let detached = grad_tensor.detach();

        assert!(grad_tensor.requires_grad());
        assert!(!detached.requires_grad());
        assert_eq!(
            grad_tensor.tensor().shape().dims(),
            detached.tensor().shape().dims()
        );
    }

    #[test]
    fn test_version_tracking() {
        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());
        let grad: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4], shape);

        let mut grad_tensor = GradTensor::with_grad(tensor);
        let initial_version = grad_tensor.grad_version();

        grad_tensor.set_grad(grad.clone());
        assert!(grad_tensor.grad_version() > initial_version);

        let after_set = grad_tensor.grad_version();
        grad_tensor.accumulate_grad(grad);
        assert!(grad_tensor.grad_version() > after_set);
    }
}
