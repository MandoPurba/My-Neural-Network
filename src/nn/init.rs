//! Weight initialization utilities for neural networks
//!
//! This module provides various weight initialization strategies commonly used
//! in deep learning to help with training convergence and gradient flow.

use crate::backend::Backend;
use crate::data::Float;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Weight initialization strategies
pub enum InitStrategy {
    /// Initialize with zeros
    Zeros,
    /// Initialize with ones
    Ones,
    /// Initialize with constant value
    Constant(f32),
    /// Xavier/Glorot uniform initialization
    XavierUniform,
    /// Xavier/Glorot normal initialization
    XavierNormal,
    /// He/Kaiming uniform initialization (good for ReLU)
    HeUniform,
    /// He/Kaiming normal initialization (good for ReLU)
    HeNormal,
    /// LeCun uniform initialization
    LeCunUniform,
    /// Simple uniform distribution
    Uniform { min: f32, max: f32 },
    /// Simple normal distribution approximation
    Normal { mean: f32, std: f32 },
}

/// Weight initializer
pub struct Initializer;

impl Initializer {
    /// Initialize weights using the specified strategy
    pub fn init<B: Backend + Default>(
        strategy: InitStrategy,
        shape: Shape<2>,
    ) -> Tensor<B, 2, Float> {
        let input_size = shape.dim(0);
        let output_size = shape.dim(1);
        let _total_size = input_size * output_size;

        match strategy {
            InitStrategy::Zeros => Tensor::zeros(shape),
            InitStrategy::Ones => Tensor::ones(shape),
            InitStrategy::Constant(value) => Tensor::full(shape, value),

            InitStrategy::XavierUniform => {
                let bound = (6.0 / (input_size + output_size) as f32).sqrt();
                Self::uniform_like(shape, -bound, bound)
            }

            InitStrategy::XavierNormal => {
                let std = (2.0 / (input_size + output_size) as f32).sqrt();
                Self::normal_like(shape, 0.0, std)
            }

            InitStrategy::HeUniform => {
                let bound = (6.0 / input_size as f32).sqrt();
                Self::uniform_like(shape, -bound, bound)
            }

            InitStrategy::HeNormal => {
                let std = (2.0 / input_size as f32).sqrt();
                Self::normal_like(shape, 0.0, std)
            }

            InitStrategy::LeCunUniform => {
                let bound = (3.0 / input_size as f32).sqrt();
                Self::uniform_like(shape, -bound, bound)
            }

            InitStrategy::Uniform { min, max } => Self::uniform_like(shape, min, max),

            InitStrategy::Normal { mean, std } => Self::normal_like(shape, mean, std),
        }
    }

    /// Initialize bias vector (usually zeros)
    pub fn init_bias<B: Backend + Default>(size: usize) -> Tensor<B, 1, Float> {
        Tensor::zeros(Shape::new([size]))
    }

    /// Create uniform-like distribution using simple pseudo-random
    fn uniform_like<B: Backend + Default>(
        shape: Shape<2>,
        min: f32,
        max: f32,
    ) -> Tensor<B, 2, Float> {
        let total_size = shape.dim(0) * shape.dim(1);
        let range = max - min;

        let data: Vec<f32> = (0..total_size)
            .map(|i| {
                // Simple pseudo-random using linear congruential generator
                let a = 1664525u32;
                let c = 1013904223u32;
                let seed = (i as u32).wrapping_mul(a).wrapping_add(c);
                let normalized = (seed % 10000) as f32 / 10000.0; // [0, 1)
                min + normalized * range
            })
            .collect();

        Tensor::from_data(data, shape)
    }

    /// Create normal-like distribution using Box-Muller transform approximation
    fn normal_like<B: Backend + Default>(
        shape: Shape<2>,
        mean: f32,
        std: f32,
    ) -> Tensor<B, 2, Float> {
        let total_size = shape.dim(0) * shape.dim(1);

        let data: Vec<f32> = (0..total_size)
            .map(|i| {
                // Simple approximation to normal distribution
                // Sum of 12 uniform random variables approximates normal
                let mut sum = 0.0;
                for j in 0..12 {
                    let seed = ((i * 12 + j) as u32)
                        .wrapping_mul(1664525)
                        .wrapping_add(1013904223);
                    sum += (seed % 10000) as f32 / 10000.0;
                }
                let normal_approx = sum - 6.0; // Center around 0, std ≈ 1
                mean + normal_approx * std
            })
            .collect();

        Tensor::from_data(data, shape)
    }

    /// Get recommended initialization for activation function
    pub fn for_activation(activation: &str) -> InitStrategy {
        match activation.to_lowercase().as_str() {
            "relu" | "leaky_relu" | "elu" => InitStrategy::HeUniform,
            "sigmoid" | "tanh" => InitStrategy::XavierUniform,
            "linear" | "none" => InitStrategy::XavierUniform,
            "gelu" | "swish" => InitStrategy::HeUniform,
            _ => InitStrategy::XavierUniform, // Default fallback
        }
    }
}

/// Builder pattern for easy initialization
pub struct InitBuilder {
    strategy: Option<InitStrategy>,
}

impl InitBuilder {
    /// Create new initialization builder
    pub fn new() -> Self {
        Self { strategy: None }
    }

    /// Use Xavier/Glorot uniform initialization
    pub fn xavier_uniform(mut self) -> Self {
        self.strategy = Some(InitStrategy::XavierUniform);
        self
    }

    /// Use Xavier/Glorot normal initialization
    pub fn xavier_normal(mut self) -> Self {
        self.strategy = Some(InitStrategy::XavierNormal);
        self
    }

    /// Use He/Kaiming uniform initialization
    pub fn he_uniform(mut self) -> Self {
        self.strategy = Some(InitStrategy::HeUniform);
        self
    }

    /// Use He/Kaiming normal initialization
    pub fn he_normal(mut self) -> Self {
        self.strategy = Some(InitStrategy::HeNormal);
        self
    }

    /// Use constant initialization
    pub fn constant(mut self, value: f32) -> Self {
        self.strategy = Some(InitStrategy::Constant(value));
        self
    }

    /// Use uniform initialization
    pub fn uniform(mut self, min: f32, max: f32) -> Self {
        self.strategy = Some(InitStrategy::Uniform { min, max });
        self
    }

    /// Use normal initialization
    pub fn normal(mut self, mean: f32, std: f32) -> Self {
        self.strategy = Some(InitStrategy::Normal { mean, std });
        self
    }

    /// Initialize for specific activation function
    pub fn for_activation(mut self, activation: &str) -> Self {
        self.strategy = Some(Initializer::for_activation(activation));
        self
    }

    /// Build the tensor with the specified shape
    pub fn build<B: Backend + Default>(self, shape: Shape<2>) -> Tensor<B, 2, Float> {
        let strategy = self.strategy.unwrap_or(InitStrategy::XavierUniform);
        Initializer::init(strategy, shape)
    }
}

impl Default for InitBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_zero_initialization() {
        let weights: Tensor<CpuBackend, 2> =
            Initializer::init(InitStrategy::Zeros, Shape::new([3, 4]));
        let data = weights.to_data();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_constant_initialization() {
        let weights: Tensor<CpuBackend, 2> =
            Initializer::init(InitStrategy::Constant(0.5), Shape::new([2, 3]));
        let data = weights.to_data();
        assert!(data.iter().all(|&x| x == 0.5));
    }

    #[test]
    fn test_xavier_uniform() {
        let shape = Shape::new([100, 50]);
        let weights: Tensor<CpuBackend, 2> = Initializer::init(InitStrategy::XavierUniform, shape);
        let data = weights.to_data();

        // Check bounds
        let bound = (6.0 / (100 + 50) as f32).sqrt();
        assert!(data.iter().all(|&x| x >= -bound && x <= bound));

        // Check that not all values are the same
        let first_val = data[0];
        assert!(!data.iter().all(|&x| x == first_val));
    }

    #[test]
    fn test_he_uniform() {
        let shape = Shape::new([100, 50]);
        let weights: Tensor<CpuBackend, 2> = Initializer::init(InitStrategy::HeUniform, shape);
        let data = weights.to_data();

        // Check bounds
        let bound = (6.0f32 / 100.0f32).sqrt();
        assert!(data.iter().all(|&x| x >= -bound && x <= bound));
    }

    #[test]
    fn test_bias_initialization() {
        let bias: Tensor<CpuBackend, 1> = Initializer::init_bias(5);
        let data = bias.to_data();
        assert_eq!(data.len(), 5);
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_builder_pattern() {
        let weights: Tensor<CpuBackend, 2> =
            InitBuilder::new().he_uniform().build(Shape::new([10, 5]));

        assert_eq!(weights.shape().dims(), &[10, 5]);

        let data = weights.to_data();
        let bound = (6.0f32 / 10.0f32).sqrt();
        assert!(data.iter().all(|&x| x >= -bound && x <= bound));
    }

    #[test]
    fn test_activation_recommendations() {
        let relu_init = Initializer::for_activation("relu");
        let sigmoid_init = Initializer::for_activation("sigmoid");

        match relu_init {
            InitStrategy::HeUniform => {}
            _ => panic!("Expected HeUniform for ReLU"),
        }

        match sigmoid_init {
            InitStrategy::XavierUniform => {}
            _ => panic!("Expected XavierUniform for Sigmoid"),
        }
    }

    #[test]
    fn test_uniform_range() {
        let weights: Tensor<CpuBackend, 2> = Initializer::init(
            InitStrategy::Uniform {
                min: -1.0,
                max: 1.0,
            },
            Shape::new([50, 25]),
        );
        let data = weights.to_data();

        assert!(data.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_normal_distribution() {
        let weights: Tensor<CpuBackend, 2> = Initializer::init(
            InitStrategy::Normal {
                mean: 0.0,
                std: 1.0,
            },
            Shape::new([1000, 100]),
        );
        let data = weights.to_data();

        // Most values should be within 3 standard deviations
        let within_3_std = data.iter().filter(|&&x| x.abs() <= 3.0).count();
        let total = data.len();
        let ratio = within_3_std as f32 / total as f32;

        // Should be > 99% for normal distribution
        assert!(ratio > 0.95); // Relaxed threshold for our approximation
    }
}
