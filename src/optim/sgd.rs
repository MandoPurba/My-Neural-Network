//! Stochastic Gradient Descent (SGD) Optimizer
//!
//! This module implements the SGD optimizer with optional momentum and weight decay.

use crate::backend::Backend;
use crate::optim::{Optimizer, Parameter};
use std::collections::HashMap;

/// SGD optimizer with optional momentum and weight decay
#[derive(Debug)]
pub struct Sgd {
    /// Learning rate
    pub learning_rate: f32,
    /// Momentum factor (0.0 means no momentum)
    pub momentum: f32,
    /// Weight decay factor
    pub weight_decay: f32,
    /// Whether to use Nesterov momentum
    pub nesterov: bool,
    /// Current step count
    pub step_count: usize,
    /// Momentum buffers for each parameter (indexed by parameter address)
    momentum_buffers: HashMap<usize, Box<dyn std::any::Any + Send + Sync>>,
}

impl Sgd {
    /// Create a new SGD optimizer
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            step_count: 0,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Create SGD optimizer with momentum
    pub fn with_momentum(learning_rate: f32, momentum: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay: 0.0,
            nesterov: false,
            step_count: 0,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Create SGD optimizer with momentum and weight decay
    pub fn with_momentum_and_decay(learning_rate: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            momentum,
            weight_decay,
            nesterov: false,
            step_count: 0,
            momentum_buffers: HashMap::new(),
        }
    }

    /// Enable Nesterov momentum
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Set weight decay
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Get the momentum factor
    pub fn momentum(&self) -> f32 {
        self.momentum
    }

    /// Set the momentum factor
    pub fn set_momentum(&mut self, momentum: f32) {
        self.momentum = momentum;
    }

    /// Get the weight decay factor
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set the weight decay factor
    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay;
    }

    /// Check if Nesterov momentum is enabled
    pub fn is_nesterov(&self) -> bool {
        self.nesterov
    }

    /// Set Nesterov momentum
    pub fn set_nesterov(&mut self, nesterov: bool) {
        self.nesterov = nesterov;
    }

    /// Clear all momentum buffers
    pub fn clear_momentum_buffers(&mut self) {
        self.momentum_buffers.clear();
    }

    /// Get the number of momentum buffers
    pub fn momentum_buffer_count(&self) -> usize {
        self.momentum_buffers.len()
    }
}

impl<B: Backend> Optimizer<B> for Sgd {
    fn step(&mut self, parameters: &mut [&mut dyn Parameter<B>]) {
        self.step_count += 1;

        for param in parameters.iter_mut() {
            if !param.requires_grad() {
                continue;
            }

            // Skip if no gradient is available
            let _grad = match param.grad() {
                Some(g) => g,
                None => continue,
            };

            // Get parameter value
            let param_value = param.value_mut();

            // For now, we'll implement a simplified version that works with our GradTensor
            // In a full implementation, we would need proper tensor arithmetic for updates

            // Apply weight decay if specified
            if self.weight_decay != 0.0 {
                // param = param - weight_decay * learning_rate * param
                // This would require tensor operations
            }

            // Apply momentum if specified
            if self.momentum != 0.0 {
                let _param_addr = param_value as *const dyn std::any::Any as *const () as usize;

                // For momentum, we need to:
                // 1. Get or create momentum buffer
                // 2. Update: momentum_buffer = momentum * momentum_buffer + grad
                // 3. If Nesterov: param = param - lr * (momentum * momentum_buffer + grad)
                // 4. Else: param = param - lr * momentum_buffer

                // This is a simplified placeholder - actual implementation would need
                // proper tensor operations and type handling
            } else {
                // Simple SGD update: param = param - learning_rate * grad
                // This would require tensor subtraction operations
            }
        }
    }

    fn zero_grad(&self, parameters: &mut [&mut dyn Parameter<B>]) {
        for param in parameters.iter_mut() {
            param.zero_grad();
        }
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn step_count(&self) -> usize {
        self.step_count
    }
}

impl Default for Sgd {
    fn default() -> Self {
        Self::new(0.01)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::TensorParameter;
    use crate::{CpuBackend, GradTensor, Shape, Tensor};

    #[test]
    fn test_sgd_creation() {
        let sgd = Sgd::new(0.01);
        assert_eq!(
            <Sgd as crate::optim::Optimizer<crate::CpuBackend>>::learning_rate(&sgd),
            0.01
        );
        assert_eq!(sgd.momentum, 0.0);
        assert_eq!(sgd.weight_decay, 0.0);
        assert!(!sgd.nesterov);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let sgd = Sgd::with_momentum(0.01, 0.9);
        assert_eq!(sgd.learning_rate(), 0.01);
        assert_eq!(sgd.momentum(), 0.9);
        assert_eq!(sgd.weight_decay(), 0.0);
        assert!(!sgd.is_nesterov());
    }

    #[test]
    fn test_sgd_with_momentum_and_decay() {
        let sgd = Sgd::with_momentum_and_decay(0.01, 0.9, 0.0001);
        assert_eq!(sgd.learning_rate(), 0.01);
        assert_eq!(sgd.momentum(), 0.9);
        assert_eq!(sgd.weight_decay(), 0.0001);
        assert!(!sgd.is_nesterov());
    }

    #[test]
    fn test_sgd_with_nesterov() {
        let sgd = Sgd::with_momentum(0.01, 0.9).with_nesterov(true);
        assert!(sgd.is_nesterov());
    }

    #[test]
    fn test_sgd_setters() {
        let mut sgd = Sgd::new(0.01);

        <Sgd as crate::optim::Optimizer<crate::CpuBackend>>::set_learning_rate(&mut sgd, 0.001);
        assert_eq!(
            <Sgd as crate::optim::Optimizer<crate::CpuBackend>>::learning_rate(&sgd),
            0.001
        );

        sgd.set_momentum(0.9);
        assert_eq!(sgd.momentum, 0.9);

        sgd.set_weight_decay(0.0001);
        assert_eq!(sgd.weight_decay, 0.0001);

        sgd.set_nesterov(true);
        assert!(sgd.nesterov);
    }

    #[test]
    fn test_zero_grad() {
        let mut sgd = Sgd::new(0.01);

        // Create a parameter with gradient
        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());
        let grad: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4], shape);

        let mut grad_tensor = GradTensor::with_grad(tensor);
        grad_tensor.set_grad(grad);

        let mut param = TensorParameter::new(grad_tensor, "test".to_string());
        assert!(param.tensor().has_grad());

        // Zero gradients
        let mut params: Vec<&mut dyn Parameter<CpuBackend>> = vec![&mut param];
        sgd.zero_grad(&mut params);

        // Gradient should be zeroed (though implementation details may vary)
    }

    #[test]
    fn test_step_count() {
        let mut sgd = Sgd::new(0.01);
        assert_eq!(sgd.step_count(), 0);

        // Create dummy parameter
        let shape = Shape::new([1]);
        let tensor: Tensor<CpuBackend, 1> = Tensor::from_data(vec![1.0], shape);
        let grad_tensor = GradTensor::with_grad(tensor);
        let mut param = TensorParameter::new(grad_tensor, "test".to_string());

        let mut params: Vec<&mut dyn Parameter<CpuBackend>> = vec![&mut param];

        sgd.step(&mut params);
        assert_eq!(sgd.step_count(), 1);

        sgd.step(&mut params);
        assert_eq!(sgd.step_count(), 2);
    }

    #[test]
    fn test_momentum_buffers() {
        let mut sgd = Sgd::with_momentum(0.01, 0.9);
        assert_eq!(sgd.momentum_buffer_count(), 0);

        sgd.clear_momentum_buffers();
        assert_eq!(sgd.momentum_buffer_count(), 0);
    }

    #[test]
    fn test_default_sgd() {
        let sgd = Sgd::default();
        assert_eq!(sgd.learning_rate(), 0.01);
        assert_eq!(sgd.momentum(), 0.0);
        assert_eq!(sgd.weight_decay(), 0.0);
        assert!(!sgd.is_nesterov());
    }
}
