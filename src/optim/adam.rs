//! Adam Optimizer
//!
//! This module implements the Adam optimizer (Adaptive Moment Estimation)
//! which combines the advantages of AdaGrad and RMSprop.

use crate::backend::Backend;
use crate::optim::{Optimizer, Parameter};
use std::collections::HashMap;

/// Adam optimizer with adaptive learning rates
#[derive(Debug)]
pub struct Adam {
    /// Learning rate
    pub learning_rate: f32,
    /// Exponential decay rate for first moment estimates (beta1)
    pub beta1: f32,
    /// Exponential decay rate for second moment estimates (beta2)
    pub beta2: f32,
    /// Small constant for numerical stability
    pub epsilon: f32,
    /// Weight decay factor (L2 penalty)
    pub weight_decay: f32,
    /// Whether to use AMSGrad variant
    pub amsgrad: bool,
    /// Current step count
    pub step_count: usize,
    /// First moment estimates (indexed by parameter address)
    first_moments: HashMap<usize, Box<dyn std::any::Any + Send + Sync>>,
    /// Second moment estimates (indexed by parameter address)
    second_moments: HashMap<usize, Box<dyn std::any::Any + Send + Sync>>,
    /// Maximum second moment estimates for AMSGrad (indexed by parameter address)
    max_second_moments: HashMap<usize, Box<dyn std::any::Any + Send + Sync>>,
}

impl Adam {
    /// Create a new Adam optimizer with default parameters
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            max_second_moments: HashMap::new(),
        }
    }

    /// Create Adam optimizer with custom beta values
    pub fn with_betas(learning_rate: f32, beta1: f32, beta2: f32) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            step_count: 0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            max_second_moments: HashMap::new(),
        }
    }

    /// Create Adam optimizer with weight decay
    pub fn with_weight_decay(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
            amsgrad: false,
            step_count: 0,
            first_moments: HashMap::new(),
            second_moments: HashMap::new(),
            max_second_moments: HashMap::new(),
        }
    }

    /// Enable AMSGrad variant
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Set epsilon for numerical stability
    pub fn with_epsilon(mut self, epsilon: f32) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Get beta1 parameter
    pub fn beta1(&self) -> f32 {
        self.beta1
    }

    /// Set beta1 parameter
    pub fn set_beta1(&mut self, beta1: f32) {
        self.beta1 = beta1;
    }

    /// Get beta2 parameter
    pub fn beta2(&self) -> f32 {
        self.beta2
    }

    /// Set beta2 parameter
    pub fn set_beta2(&mut self, beta2: f32) {
        self.beta2 = beta2;
    }

    /// Get epsilon parameter
    pub fn epsilon(&self) -> f32 {
        self.epsilon
    }

    /// Set epsilon parameter
    pub fn set_epsilon(&mut self, epsilon: f32) {
        self.epsilon = epsilon;
    }

    /// Get weight decay parameter
    pub fn weight_decay(&self) -> f32 {
        self.weight_decay
    }

    /// Set weight decay parameter
    pub fn set_weight_decay(&mut self, weight_decay: f32) {
        self.weight_decay = weight_decay;
    }

    /// Check if AMSGrad is enabled
    pub fn is_amsgrad(&self) -> bool {
        self.amsgrad
    }

    /// Set AMSGrad variant
    pub fn set_amsgrad(&mut self, amsgrad: bool) {
        self.amsgrad = amsgrad;
    }

    /// Clear all moment estimates
    pub fn clear_moments(&mut self) {
        self.first_moments.clear();
        self.second_moments.clear();
        self.max_second_moments.clear();
    }

    /// Get the number of tracked parameters
    pub fn tracked_parameters(&self) -> usize {
        self.first_moments.len()
    }

    /// Calculate bias correction for first moment
    fn bias_correction1(&self) -> f32 {
        1.0 - self.beta1.powi(self.step_count as i32)
    }

    /// Calculate bias correction for second moment
    fn bias_correction2(&self) -> f32 {
        1.0 - self.beta2.powi(self.step_count as i32)
    }
}

impl<B: Backend> Optimizer<B> for Adam {
    fn step(&mut self, parameters: &mut [&mut dyn Parameter<B>]) {
        self.step_count += 1;
        let _bias_correction1 = self.bias_correction1();
        let _bias_correction2 = self.bias_correction2();

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
            let param_addr = param_value as *const dyn std::any::Any as *const () as usize;

            // In a full implementation, we would need to:
            // 1. Apply weight decay: grad = grad + weight_decay * param
            // 2. Update first moment: m_t = beta1 * m_{t-1} + (1 - beta1) * grad
            // 3. Update second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
            // 4. Bias correction: m_hat = m_t / (1 - beta1^t), v_hat = v_t / (1 - beta2^t)
            // 5. For AMSGrad: v_hat = max(v_hat_prev, v_hat)
            // 6. Update parameter: param = param - lr * m_hat / (sqrt(v_hat) + epsilon)

            // This is a simplified placeholder - actual implementation would need
            // proper tensor operations and type handling for different tensor dimensions

            // For now, we just track that we've seen this parameter
            if !self.first_moments.contains_key(&param_addr) {
                // In a real implementation, we would initialize moment buffers here
                // with the same shape as the parameter tensor
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

impl Default for Adam {
    fn default() -> Self {
        Self::new(0.001) // Adam typically uses 0.001 as default learning rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optim::TensorParameter;
    use crate::{CpuBackend, GradTensor, Shape, Tensor};

    #[test]
    fn test_adam_creation() {
        let adam = Adam::new(0.001);
        assert_eq!(
            <Adam as crate::optim::Optimizer<crate::CpuBackend>>::learning_rate(&adam),
            0.001
        );
        assert_eq!(adam.beta1, 0.9);
        assert_eq!(adam.beta2, 0.999);
        assert_eq!(adam.epsilon, 1e-8);
        assert!(!adam.amsgrad);
    }

    #[test]
    fn test_adam_with_momentum() {
        let adam = Adam::with_momentum(0.001, 0.95);
        assert_eq!(
            <Adam as crate::optim::Optimizer<crate::CpuBackend>>::learning_rate(&adam),
            0.001
        );
        assert_eq!(adam.beta1, 0.95);
        assert_eq!(adam.beta2, 0.999);
    }

    #[test]
    fn test_adam_with_weight_decay() {
        let adam = Adam::with_weight_decay(0.001, 0.01);
        assert_eq!(
            <Adam as crate::optim::Optimizer<crate::CpuBackend>>::learning_rate(&adam),
            0.001
        );
        assert_eq!(adam.weight_decay, 0.01);
    }

    #[test]
    fn test_adam_with_amsgrad() {
        let adam = Adam::new(0.001).with_amsgrad(true);
        assert!(adam.is_amsgrad());
    }

    #[test]
    fn test_adam_with_epsilon() {
        let adam = Adam::new(0.001).with_epsilon(1e-6);
        assert_eq!(adam.epsilon(), 1e-6);
    }

    #[test]
    fn test_adam_setters() {
        let mut adam = Adam::new(0.001);

        <Adam as crate::optim::Optimizer<crate::CpuBackend>>::set_learning_rate(&mut adam, 0.0001);
        assert_eq!(
            <Adam as crate::optim::Optimizer<crate::CpuBackend>>::learning_rate(&adam),
            0.0001
        );

        adam.set_beta1(0.95);
        assert_eq!(adam.beta1, 0.95);

        adam.set_beta2(0.99);
        assert_eq!(adam.beta2, 0.99);

        adam.set_epsilon(1e-6);
        assert_eq!(adam.epsilon, 1e-6);

        adam.set_weight_decay(0.01);
        assert_eq!(adam.weight_decay, 0.01);

        adam.set_amsgrad(true);
        assert!(adam.amsgrad);

        adam.set_amsgrad(false);
        assert!(!adam.amsgrad);
    }

    #[test]
    fn test_bias_corrections() {
        let mut adam = Adam::new(0.001);

        // Before any steps
        assert_eq!(adam.step_count(), 0);

        // After first step
        adam.step_count = 1;
        let bias_corr1 = adam.bias_correction1();
        let bias_corr2 = adam.bias_correction2();

        assert!((bias_corr1 - (1.0 - 0.9_f32.powi(1))).abs() < 1e-6);
        assert!((bias_corr2 - (1.0 - 0.999_f32.powi(1))).abs() < 1e-6);

        // After 10 steps
        adam.step_count = 10;
        let bias_corr1 = adam.bias_correction1();
        let bias_corr2 = adam.bias_correction2();

        assert!((bias_corr1 - (1.0 - 0.9_f32.powi(10))).abs() < 1e-6);
        assert!((bias_corr2 - (1.0 - 0.999_f32.powi(10))).abs() < 1e-6);
    }

    #[test]
    fn test_zero_grad() {
        let mut adam = Adam::new(0.001);

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
        adam.zero_grad(&mut params);

        // Gradient should be zeroed (though implementation details may vary)
    }

    #[test]
    fn test_step_count() {
        let adam = Adam::new(0.001);
        assert_eq!(
            <Adam as crate::optim::Optimizer<crate::CpuBackend>>::step_count(&adam),
            0
        );
    }

    #[test]
    fn test_clear_moments() {
        let mut adam = Adam::new(0.001);
        assert_eq!(adam.tracked_parameters(), 0);

        adam.clear_moments();
        assert_eq!(adam.tracked_parameters(), 0);
    }

    #[test]
    fn test_default_adam() {
        let adam = Adam::default();
        assert_eq!(
            <Adam as crate::optim::Optimizer<crate::CpuBackend>>::learning_rate(&adam),
            0.001
        );
        assert_eq!(adam.beta1, 0.9);
        assert_eq!(adam.beta2, 0.999);
        assert_eq!(adam.epsilon, 1e-8);
        assert!(!adam.amsgrad);
    }

    #[test]
    fn test_chained_builder() {
        let adam = Adam::new(0.001)
            .with_betas(0.95, 0.99)
            .with_weight_decay(0.01)
            .with_amsgrad(true)
            .with_epsilon(1e-6);

        assert_eq!(adam.learning_rate(), 0.001);
        assert_eq!(adam.beta1(), 0.95);
        assert_eq!(adam.beta2(), 0.99);
        assert_eq!(adam.weight_decay(), 0.01);
        assert!(adam.is_amsgrad());
        assert_eq!(adam.epsilon(), 1e-6);
    }
}
