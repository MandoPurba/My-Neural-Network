//! Test utilities for automatic differentiation
//!
//! This module provides utilities for testing gradient computations,
//! including numerical gradient checking using finite differences.

use crate::backend::Backend;

use crate::shape::Shape;
use crate::tensor::Tensor;

/// Numerical gradient checking using finite differences
pub struct NumericalGradChecker {
    /// Step size for finite differences
    pub epsilon: f64,
    /// Relative tolerance for gradient comparison
    pub rtol: f64,
    /// Absolute tolerance for gradient comparison
    pub atol: f64,
}

impl Default for NumericalGradChecker {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            rtol: 1e-3,
            atol: 1e-5,
        }
    }
}

impl NumericalGradChecker {
    /// Create a new gradient checker with custom tolerances
    pub fn new(epsilon: f64, rtol: f64, atol: f64) -> Self {
        Self {
            epsilon,
            rtol,
            atol,
        }
    }

    /// Check gradients using finite differences for a scalar function
    pub fn check_gradients<B, F, const D: usize>(
        &self,
        inputs: &[Tensor<B, D>],
        function: F,
        analytical_grads: &[Tensor<B, D>],
    ) -> Result<(), String>
    where
        B: Backend + Default,
        F: Fn(&[Tensor<B, D>]) -> Tensor<B, 1>,
    {
        if inputs.len() != analytical_grads.len() {
            return Err("Number of inputs and gradients must match".to_string());
        }

        for (i, (_input, analytical_grad)) in inputs.iter().zip(analytical_grads.iter()).enumerate()
        {
            let numerical_grad = self.compute_numerical_gradient(inputs, i, &function)?;

            if !self.gradients_close(&numerical_grad, analytical_grad) {
                return Err(format!(
                    "Gradient mismatch for input {}: numerical={:?}, analytical={:?}",
                    i,
                    numerical_grad.to_data(),
                    analytical_grad.to_data()
                ));
            }
        }

        Ok(())
    }

    /// Compute numerical gradient for one input using finite differences
    fn compute_numerical_gradient<B, F, const D: usize>(
        &self,
        inputs: &[Tensor<B, D>],
        input_idx: usize,
        function: &F,
    ) -> Result<Tensor<B, D>, String>
    where
        B: Backend + Default,
        F: Fn(&[Tensor<B, D>]) -> Tensor<B, 1>,
    {
        let input = &inputs[input_idx];
        let input_data = input.to_data();
        let mut grad_data = vec![0.0; input_data.len()];

        for (j, _) in input_data.iter().enumerate() {
            // Create positive perturbation
            let mut inputs_pos = inputs.to_vec();
            let mut data_pos = inputs_pos[input_idx].to_data();
            data_pos[j] += self.epsilon as f32;
            inputs_pos[input_idx] = Tensor::from_data(data_pos, input.shape().clone());

            // Create negative perturbation
            let mut inputs_neg = inputs.to_vec();
            let mut data_neg = inputs_neg[input_idx].to_data();
            data_neg[j] -= self.epsilon as f32;
            inputs_neg[input_idx] = Tensor::from_data(data_neg, input.shape().clone());

            // Compute finite difference
            let output_pos = function(&inputs_pos);
            let output_neg = function(&inputs_neg);

            let output_pos_data = output_pos.to_data();
            let output_neg_data = output_neg.to_data();

            if output_pos_data.len() != 1 || output_neg_data.len() != 1 {
                return Err("Function must return a scalar output".to_string());
            }

            grad_data[j] = (output_pos_data[0] - output_neg_data[0]) / (2.0 * self.epsilon as f32);
        }

        Ok(Tensor::from_data(grad_data, input.shape().clone()))
    }

    /// Check if two gradients are close within tolerance
    fn gradients_close<B: Backend + Default, const D: usize>(
        &self,
        grad1: &Tensor<B, D>,
        grad2: &Tensor<B, D>,
    ) -> bool {
        let data1 = grad1.to_data();
        let data2 = grad2.to_data();

        if data1.len() != data2.len() {
            return false;
        }

        for (a, b) in data1.iter().zip(data2.iter()) {
            let diff = (a - b).abs();
            let max_val = a.abs().max(b.abs());

            if diff > self.atol as f32 + self.rtol as f32 * max_val {
                return false;
            }
        }

        true
    }
}

/// Create a simple test function for gradient checking
/// Returns x^2 + y^2 (sum of squares)
pub fn sum_of_squares<B: Backend + Default>(inputs: &[Tensor<B, 1>]) -> Tensor<B, 1> {
    let mut result = 0.0;

    for input in inputs {
        let data = input.to_data();
        for &x in &data {
            result += x * x;
        }
    }

    Tensor::from_data(vec![result], Shape::new([1]))
}

/// Create a test function that computes the dot product of two vectors
pub fn dot_product<B: Backend + Default>(inputs: &[Tensor<B, 1>]) -> Tensor<B, 1> {
    if inputs.len() != 2 {
        panic!("Dot product requires exactly 2 inputs");
    }

    let data1 = inputs[0].to_data();
    let data2 = inputs[1].to_data();

    if data1.len() != data2.len() {
        panic!("Input vectors must have the same length");
    }

    let result: f32 = data1.iter().zip(data2.iter()).map(|(a, b)| a * b).sum();

    Tensor::from_data(vec![result], Shape::new([1]))
}

/// Create a test function that computes the L2 norm squared
pub fn l2_norm_squared<B: Backend + Default>(inputs: &[Tensor<B, 1>]) -> Tensor<B, 1> {
    if inputs.len() != 1 {
        panic!("L2 norm requires exactly 1 input");
    }

    let data = inputs[0].to_data();
    let result: f32 = data.iter().map(|x| x * x).sum();

    Tensor::from_data(vec![result], Shape::new([1]))
}

/// Generate random test data for gradient checking
pub fn generate_test_data<B: Backend + Default, const D: usize>(
    shape: Shape<D>,
    num_tensors: usize,
) -> Vec<Tensor<B, D>> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut tensors = Vec::new();

    for i in 0..num_tensors {
        // Simple deterministic "random" data generation
        let mut hasher = DefaultHasher::new();
        i.hash(&mut hasher);
        let seed = hasher.finish();

        let size = shape.num_elements();
        let mut data = Vec::with_capacity(size);

        for j in 0..size {
            let mut item_hasher = DefaultHasher::new();
            (seed + j as u64).hash(&mut item_hasher);
            let val = (item_hasher.finish() % 1000) as f32 / 1000.0 - 0.5; // Range [-0.5, 0.5]
            data.push(val);
        }

        tensors.push(Tensor::from_data(data, shape.clone()));
    }

    tensors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CpuBackend, Shape};

    #[test]
    fn test_numerical_grad_checker_creation() {
        let checker = NumericalGradChecker::default();
        assert_eq!(checker.epsilon, 1e-5);
        assert_eq!(checker.rtol, 1e-3);
        assert_eq!(checker.atol, 1e-5);

        let custom_checker = NumericalGradChecker::new(1e-6, 1e-4, 1e-6);
        assert_eq!(custom_checker.epsilon, 1e-6);
        assert_eq!(custom_checker.rtol, 1e-4);
        assert_eq!(custom_checker.atol, 1e-6);
    }

    #[test]
    fn test_sum_of_squares() {
        let shape = Shape::new([2]);
        let input1 = Tensor::<CpuBackend, 1>::from_data(vec![2.0, 3.0], shape.clone());
        let input2 = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 4.0], shape.clone());

        let result = sum_of_squares(&[input1, input2]);
        let result_data = result.to_data();

        // Expected: 2^2 + 3^2 + 1^2 + 4^2 = 4 + 9 + 1 + 16 = 30
        assert_eq!(result_data, vec![30.0]);
    }

    #[test]
    fn test_dot_product() {
        let shape = Shape::new([3]);
        let input1 = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0, 3.0], shape.clone());
        let input2 = Tensor::<CpuBackend, 1>::from_data(vec![4.0, 5.0, 6.0], shape.clone());

        let result = dot_product(&[input1, input2]);
        let result_data = result.to_data();

        // Expected: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        assert_eq!(result_data, vec![32.0]);
    }

    #[test]
    fn test_l2_norm_squared() {
        let shape = Shape::new([3]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![2.0, 3.0, 6.0], shape.clone());

        let result = l2_norm_squared(&[input]);
        let result_data = result.to_data();

        // Expected: 2^2 + 3^2 + 6^2 = 4 + 9 + 36 = 49
        assert_eq!(result_data, vec![49.0]);
    }

    #[test]
    fn test_generate_test_data() {
        let shape = Shape::new([3]);
        let tensors = generate_test_data::<CpuBackend, 1>(shape.clone(), 2);

        assert_eq!(tensors.len(), 2);
        assert_eq!(tensors[0].shape(), &shape);
        assert_eq!(tensors[1].shape(), &shape);

        // Check that data is in expected range [-0.5, 0.5]
        for tensor in &tensors {
            let data = tensor.to_data();
            for &val in &data {
                assert!(val >= -0.5 && val <= 0.5);
            }
        }

        // Check that tensors are different (deterministic but different seeds)
        assert_ne!(tensors[0].to_data(), tensors[1].to_data());
    }

    #[test]
    fn test_gradients_close() {
        let checker = NumericalGradChecker::default();
        let shape = Shape::new([2]);

        let grad1 = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0], shape.clone());
        let grad2 = Tensor::<CpuBackend, 1>::from_data(vec![1.001, 2.001], shape.clone());
        let grad3 = Tensor::<CpuBackend, 1>::from_data(vec![1.1, 2.1], shape.clone());

        // Should be close within tolerance
        assert!(checker.gradients_close(&grad1, &grad2));

        // Should not be close (difference too large)
        assert!(!checker.gradients_close(&grad1, &grad3));
    }

    #[test]
    fn test_numerical_gradient_simple() {
        let checker = NumericalGradChecker::new(1e-5, 1e-2, 1e-4);
        let shape = Shape::new([1]);

        // Test function f(x) = x^2, so df/dx = 2x
        let input = Tensor::<CpuBackend, 1>::from_data(vec![3.0], shape.clone());
        let inputs = vec![input];

        let numerical_grad = checker
            .compute_numerical_gradient(&inputs, 0, &l2_norm_squared)
            .unwrap();
        let numerical_data = numerical_grad.to_data();

        // Expected gradient at x=3 is 2*3 = 6
        assert!((numerical_data[0] - 6.0).abs() < 1e-3);
    }
}
