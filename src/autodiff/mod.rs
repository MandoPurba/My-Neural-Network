//! Automatic Differentiation (Autodiff) Module
//!
//! This module implements reverse-mode automatic differentiation using a tape-based approach.
//! It tracks computational graphs during forward passes and enables gradient computation
//! during backward passes.

pub mod grad;
pub mod graph;
pub mod ops;
pub mod tape;
pub mod test_utils;

pub use grad::{GradTensor, RequiresGrad};
pub use graph::{ComputationGraph, NodeId};
pub use ops::{
    AddGradFn, AddOp, MulGradFn, MulOp, ReluGradFn, ReluOp, SigmoidGradFn, SigmoidOp, SumGradFn,
    SumOp,
};
pub use tape::{GradientTape, TapeBackend};

use crate::backend::Backend;
use crate::data::DataType;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// A tensor that can track gradients for automatic differentiation
pub type AutodiffTensor<B, const D: usize, K = crate::Float> = GradTensor<B, D, K>;

/// Trait for operations that support automatic differentiation
pub trait AutodiffOp<B: Backend> {
    /// The input tensor types for this operation
    type Input;
    /// The output tensor type for this operation
    type Output;

    /// Execute the forward pass of the operation
    fn forward(&self, input: Self::Input) -> Self::Output;

    /// Execute the backward pass of the operation
    /// Returns gradients with respect to the inputs
    fn backward(&self, grad_output: Self::Output, input: Self::Input) -> Self::Input;
}

/// Context for gradient computation
pub struct GradientContext<B: Backend> {
    /// Map from node IDs to their gradients
    pub gradients: HashMap<NodeId, Box<dyn std::any::Any>>,
    /// The computation graph
    pub graph: ComputationGraph<B>,
    /// Whether we're currently tracking gradients
    pub tracking: bool,
}

impl<B: Backend> GradientContext<B> {
    /// Create a new gradient context
    pub fn new() -> Self {
        Self {
            gradients: HashMap::new(),
            graph: ComputationGraph::new(),
            tracking: true,
        }
    }

    /// Enable gradient tracking
    pub fn enable_grad(&mut self) {
        self.tracking = true;
    }

    /// Disable gradient tracking
    pub fn disable_grad(&mut self) {
        self.tracking = false;
    }

    /// Check if gradient tracking is enabled
    pub fn is_tracking(&self) -> bool {
        self.tracking
    }

    /// Store a gradient for a node
    pub fn store_gradient<const D: usize, K: DataType>(
        &mut self,
        node_id: NodeId,
        gradient: Tensor<B, D, K>,
    ) where
        K::Primitive: 'static,
    {
        self.gradients.insert(node_id, Box::new(gradient));
    }

    /// Retrieve a gradient for a node
    pub fn get_gradient<const D: usize, K: DataType>(
        &self,
        node_id: NodeId,
    ) -> Option<&Tensor<B, D, K>>
    where
        K::Primitive: 'static,
    {
        self.gradients
            .get(&node_id)?
            .downcast_ref::<Tensor<B, D, K>>()
    }

    /// Clear all stored gradients
    pub fn clear_gradients(&mut self) {
        self.gradients.clear();
    }
}

impl<B: Backend> Default for GradientContext<B> {
    fn default() -> Self {
        Self::new()
    }
}

thread_local! {
    /// Global gradient context (thread-local)
    static GRADIENT_CONTEXT: std::cell::RefCell<Option<GradientContext<crate::CpuBackend>>> =
        std::cell::RefCell::new(None);
}

/// Execute a closure with gradient tracking enabled
pub fn with_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    GRADIENT_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        if ctx.is_none() {
            *ctx = Some(GradientContext::new());
        }
        if let Some(ref mut context) = *ctx {
            context.enable_grad();
        }
    });

    let result = f();

    GRADIENT_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        if let Some(ref mut context) = *ctx {
            context.disable_grad();
        }
    });

    result
}

/// Execute a closure with gradient tracking disabled
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    GRADIENT_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        let was_tracking = ctx.as_ref().map(|c| c.is_tracking()).unwrap_or(false);

        if ctx.is_none() {
            *ctx = Some(GradientContext::new());
        }
        if let Some(ref mut context) = *ctx {
            context.disable_grad();
        }

        let result = f();

        if let Some(ref mut context) = *ctx {
            if was_tracking {
                context.enable_grad();
            }
        }

        result
    })
}

/// Check if gradient tracking is currently enabled
pub fn is_grad_enabled() -> bool {
    GRADIENT_CONTEXT.with(|ctx| {
        ctx.borrow()
            .as_ref()
            .map(|c| c.is_tracking())
            .unwrap_or(false)
    })
}

/// Access the current gradient context
pub fn with_gradient_context<F, R>(f: F) -> R
where
    F: FnOnce(&mut GradientContext<crate::CpuBackend>) -> R,
{
    GRADIENT_CONTEXT.with(|ctx| {
        let mut ctx = ctx.borrow_mut();
        if ctx.is_none() {
            *ctx = Some(GradientContext::new());
        }
        f(ctx.as_mut().unwrap())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, Shape, Tensor};

    #[test]
    fn test_gradient_context() {
        let mut ctx = GradientContext::<CpuBackend>::new();
        assert!(ctx.is_tracking());

        ctx.disable_grad();
        assert!(!ctx.is_tracking());

        ctx.enable_grad();
        assert!(ctx.is_tracking());
    }

    #[test]
    fn test_with_grad() {
        // Initially no gradient tracking
        assert!(!is_grad_enabled());

        with_grad(|| {
            assert!(is_grad_enabled());
        });

        // Should be disabled again after the closure
        assert!(!is_grad_enabled());
    }

    #[test]
    fn test_no_grad() {
        with_grad(|| {
            assert!(is_grad_enabled());

            no_grad(|| {
                assert!(!is_grad_enabled());
            });

            // Should be enabled again after no_grad block
            assert!(is_grad_enabled());
        });
    }

    #[test]
    fn test_gradient_storage() {
        let mut ctx = GradientContext::<CpuBackend>::new();
        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);

        let node_id = NodeId::new();
        ctx.store_gradient(node_id, tensor.clone());

        let retrieved = ctx.get_gradient::<2, crate::Float>(node_id);
        assert!(retrieved.is_some());

        let retrieved = retrieved.unwrap();
        assert_eq!(retrieved.shape().dims(), &[2, 2]);
    }
}
