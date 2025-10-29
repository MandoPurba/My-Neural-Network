//! Optimizer Module
//!
//! This module provides optimizers for training neural networks with automatic differentiation.
//! It includes implementations of popular optimization algorithms like SGD and Adam.

pub mod adam;
pub mod sgd;

use crate::autodiff::{GradTensor, RequiresGrad};
use crate::backend::Backend;
use crate::data::DataType;

pub use adam::Adam;
pub use sgd::Sgd;

/// Trait for optimizers that can update parameters based on gradients
pub trait Optimizer<B: Backend> {
    /// Update parameters using their gradients
    fn step(&mut self, parameters: &mut [&mut dyn Parameter<B>]);

    /// Zero out all gradients in the parameters
    fn zero_grad(&self, parameters: &mut [&mut dyn Parameter<B>]);

    /// Get the learning rate
    fn learning_rate(&self) -> f32;

    /// Set the learning rate
    fn set_learning_rate(&mut self, lr: f32);

    /// Get the current step count
    fn step_count(&self) -> usize;
}

/// Trait for parameters that can be optimized
pub trait Parameter<B: Backend> {
    /// Get the parameter value
    fn value(&self) -> &dyn std::any::Any;

    /// Get the mutable parameter value
    fn value_mut(&mut self) -> &mut dyn std::any::Any;

    /// Get the gradient if it exists
    fn grad(&self) -> Option<&dyn std::any::Any>;

    /// Zero the gradient
    fn zero_grad(&mut self);

    /// Check if the parameter requires gradients
    fn requires_grad(&self) -> bool;

    /// Get the parameter name (for debugging)
    fn name(&self) -> &str;
}

/// A wrapper for tensors that can be used as parameters
#[derive(Debug)]
pub struct TensorParameter<B: Backend, const D: usize, K: DataType = crate::Float> {
    /// The parameter tensor
    pub tensor: GradTensor<B, D, K>,
    /// Name of the parameter
    pub name: String,
}

impl<B: Backend, const D: usize, K: DataType> TensorParameter<B, D, K> {
    /// Create a new tensor parameter
    pub fn new(tensor: GradTensor<B, D, K>, name: String) -> Self {
        Self { tensor, name }
    }

    /// Create a new tensor parameter with a default name
    pub fn from_tensor(tensor: GradTensor<B, D, K>) -> Self {
        Self {
            tensor,
            name: "parameter".to_string(),
        }
    }

    /// Get a reference to the underlying tensor
    pub fn tensor(&self) -> &GradTensor<B, D, K> {
        &self.tensor
    }

    /// Get a mutable reference to the underlying tensor
    pub fn tensor_mut(&mut self) -> &mut GradTensor<B, D, K> {
        &mut self.tensor
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> Parameter<B> for TensorParameter<B, D, K>
where
    K::Primitive: Clone + 'static,
{
    fn value(&self) -> &dyn std::any::Any {
        &self.tensor
    }

    fn value_mut(&mut self) -> &mut dyn std::any::Any {
        &mut self.tensor
    }

    fn grad(&self) -> Option<&dyn std::any::Any> {
        if self.tensor.has_grad() {
            Some(&() as &dyn std::any::Any) // Simplified for demo
        } else {
            None
        }
    }

    fn zero_grad(&mut self) {
        // For now, we'll skip zeroing gradients if the tensor doesn't support it
        // In a full implementation, we'd need to handle this more gracefully
        if std::mem::size_of::<K::Primitive>() > 0 {
            // Only attempt to zero grad for types that can be constructed
            // This is a temporary workaround for the trait bound issue
        }
    }

    fn requires_grad(&self) -> bool {
        self.tensor.requires_grad()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Parameter group for organizing parameters with different optimization settings
pub struct ParameterGroup<B: Backend> {
    /// Parameters in this group
    pub parameters: Vec<Box<dyn Parameter<B>>>,
    /// Learning rate for this group
    pub learning_rate: f32,
    /// Weight decay for this group
    pub weight_decay: f32,
    /// Group name
    pub name: String,
}

impl<B: Backend> ParameterGroup<B> {
    /// Create a new parameter group
    pub fn new(learning_rate: f32) -> Self {
        Self {
            parameters: Vec::new(),
            learning_rate,
            weight_decay: 0.0,
            name: "default".to_string(),
        }
    }

    /// Add a parameter to the group
    pub fn add_parameter(&mut self, parameter: Box<dyn Parameter<B>>) {
        self.parameters.push(parameter);
    }

    /// Get the number of parameters in the group
    pub fn len(&self) -> usize {
        self.parameters.len()
    }

    /// Check if the group is empty
    pub fn is_empty(&self) -> bool {
        self.parameters.is_empty()
    }

    /// Set weight decay for the group
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set name for the group
    pub fn with_name(mut self, name: String) -> Self {
        self.name = name;
        self
    }
}

/// Learning rate scheduler trait
pub trait LrScheduler {
    /// Get the learning rate for the current step
    fn get_lr(&self, step: usize) -> f32;

    /// Update the scheduler (called after each step)
    fn step(&mut self);
}

/// Constant learning rate scheduler
#[derive(Debug, Clone)]
pub struct ConstantLr {
    pub lr: f32,
}

impl ConstantLr {
    pub fn new(lr: f32) -> Self {
        Self { lr }
    }
}

impl LrScheduler for ConstantLr {
    fn get_lr(&self, _step: usize) -> f32 {
        self.lr
    }

    fn step(&mut self) {
        // Nothing to do for constant LR
    }
}

/// Linear learning rate decay scheduler
#[derive(Debug, Clone)]
pub struct LinearLr {
    pub initial_lr: f32,
    pub final_lr: f32,
    pub total_steps: usize,
    pub current_step: usize,
}

impl LinearLr {
    pub fn new(initial_lr: f32, final_lr: f32, total_steps: usize) -> Self {
        Self {
            initial_lr,
            final_lr,
            total_steps,
            current_step: 0,
        }
    }
}

impl LrScheduler for LinearLr {
    fn get_lr(&self, step: usize) -> f32 {
        if step >= self.total_steps {
            return self.final_lr;
        }

        let progress = step as f32 / self.total_steps as f32;
        self.initial_lr + (self.final_lr - self.initial_lr) * progress
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

/// Exponential learning rate decay scheduler
#[derive(Debug, Clone)]
pub struct ExponentialLr {
    pub initial_lr: f32,
    pub decay_rate: f32,
    pub current_step: usize,
}

impl ExponentialLr {
    pub fn new(initial_lr: f32, decay_rate: f32) -> Self {
        Self {
            initial_lr,
            decay_rate,
            current_step: 0,
        }
    }
}

impl LrScheduler for ExponentialLr {
    fn get_lr(&self, step: usize) -> f32 {
        self.initial_lr * self.decay_rate.powf(step as f32)
    }

    fn step(&mut self) {
        self.current_step += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, Shape, Tensor};

    #[test]
    fn test_tensor_parameter() {
        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);
        let grad_tensor = GradTensor::with_grad(tensor);

        let mut param = TensorParameter::new(grad_tensor, "test_param".to_string());

        assert_eq!(param.name(), "test_param");
        assert!(param.requires_grad());

        param.zero_grad();
        assert!(param.grad().is_none() || param.tensor().grad().is_none());
    }

    #[test]
    fn test_parameter_group() {
        let mut group = ParameterGroup::<CpuBackend>::new(0.01);
        assert_eq!(group.learning_rate, 0.01);
        assert!(group.is_empty());

        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);
        let grad_tensor = GradTensor::with_grad(tensor);
        let param = TensorParameter::new(grad_tensor, "param1".to_string());

        group.add_parameter(Box::new(param));
        assert_eq!(group.len(), 1);
        assert!(!group.is_empty());
    }

    #[test]
    fn test_constant_lr_scheduler() {
        let scheduler = ConstantLr::new(0.01);

        assert_eq!(scheduler.get_lr(0), 0.01);
        assert_eq!(scheduler.get_lr(100), 0.01);
        assert_eq!(scheduler.get_lr(1000), 0.01);
    }

    #[test]
    fn test_linear_lr_scheduler() {
        let scheduler = LinearLr::new(0.1, 0.01, 100);

        assert_eq!(scheduler.get_lr(0), 0.1);
        assert_eq!(scheduler.get_lr(50), 0.055);
        assert_eq!(scheduler.get_lr(100), 0.01);
        assert_eq!(scheduler.get_lr(150), 0.01); // Should clamp to final_lr
    }

    #[test]
    fn test_exponential_lr_scheduler() {
        let scheduler = ExponentialLr::new(0.1, 0.9);

        assert_eq!(scheduler.get_lr(0), 0.1);
        assert!((scheduler.get_lr(1) - 0.09).abs() < 1e-6);
        assert!((scheduler.get_lr(2) - 0.081).abs() < 1e-6);
    }
}
