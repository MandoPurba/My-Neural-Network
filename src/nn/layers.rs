//! Neural Network Layers
//!
//! This module provides common neural network layers that can be used to build
//! deep learning models with Mini-Burn framework.

use crate::backend::Backend;
use crate::data::{DataType, Element, Float};
use crate::nn::init::{InitBuilder, Initializer};
use crate::nn::Module;
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Linear/Dense layer
pub struct Linear<B: Backend> {
    weights: Tensor<B, 2, Float>,
    bias: Tensor<B, 1, Float>,
    input_size: usize,
    output_size: usize,
}

impl<B: Backend + Default> Linear<B>
where
    <Float as DataType>::Primitive: Element,
{
    /// Create new linear layer with He initialization (good for ReLU)
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights = InitBuilder::new()
            .he_uniform()
            .build(Shape::new([input_size, output_size]));
        let bias = Initializer::init_bias(output_size);

        Self {
            weights,
            bias,
            input_size,
            output_size,
        }
    }

    /// Create linear layer with custom weight initialization
    pub fn with_init(input_size: usize, output_size: usize, init: InitBuilder) -> Self {
        let weights = init.build(Shape::new([input_size, output_size]));
        let bias = Initializer::init_bias(output_size);

        Self {
            weights,
            bias,
            input_size,
            output_size,
        }
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get output size
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Get weights reference
    pub fn weights(&self) -> &Tensor<B, 2, Float> {
        &self.weights
    }

    /// Get bias reference
    pub fn bias(&self) -> &Tensor<B, 1, Float> {
        &self.bias
    }
}

impl<B: Backend + Default> Module<B> for Linear<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        // input shape: [batch_size, input_size]
        // weights shape: [input_size, output_size]
        // output shape: [batch_size, output_size]

        let linear_output = input.matmul(&self.weights);

        // Add bias to each sample in batch
        let batch_size = input.shape().dim(0);
        let linear_data = linear_output.to_data();
        let bias_data = self.bias.to_data();

        let mut output_data = vec![0.0; batch_size * self.output_size];
        for i in 0..batch_size {
            for j in 0..self.output_size {
                output_data[i * self.output_size + j] =
                    linear_data[i * self.output_size + j] + bias_data[j];
            }
        }

        Tensor::from_data(output_data, Shape::new([batch_size, self.output_size]))
    }
}

/// ReLU activation layer
pub struct ReLU<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> ReLU<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Module<B> for ReLU<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.relu()
    }
}

impl<B: Backend> Default for ReLU<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Sigmoid activation layer
pub struct Sigmoid<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Sigmoid<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Module<B> for Sigmoid<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.sigmoid()
    }
}

impl<B: Backend> Default for Sigmoid<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Tanh activation layer
pub struct Tanh<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Tanh<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Module<B> for Tanh<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.tanh()
    }
}

impl<B: Backend> Default for Tanh<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Softmax activation layer
pub struct Softmax<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Softmax<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Module<B> for Softmax<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.softmax()
    }
}

impl<B: Backend> Default for Softmax<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Leaky ReLU activation layer
pub struct LeakyReLU<B: Backend> {
    alpha: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> LeakyReLU<B> {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_default_alpha() -> Self {
        Self::new(0.01)
    }
}

impl<B: Backend + Default> Module<B> for LeakyReLU<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.leaky_relu(self.alpha)
    }
}

/// GELU activation layer
pub struct GELU<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> GELU<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Module<B> for GELU<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.gelu()
    }
}

impl<B: Backend> Default for GELU<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Swish/SiLU activation layer
pub struct Swish<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Swish<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Module<B> for Swish<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.swish()
    }
}

impl<B: Backend> Default for Swish<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// ELU activation layer
pub struct ELU<B: Backend> {
    alpha: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> ELU<B> {
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_default_alpha() -> Self {
        Self::new(1.0)
    }
}

impl<B: Backend + Default> Module<B> for ELU<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        input.elu(self.alpha)
    }
}

/// Dropout layer (simplified, always in eval mode for this framework)
pub struct Dropout<B: Backend> {
    _rate: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Dropout<B> {
    pub fn new(rate: f32) -> Self {
        Self {
            _rate: rate,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Module<B> for Dropout<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        // In inference mode, dropout is disabled
        input
    }
}

/// Flatten layer - converts any tensor to 2D batch format
pub struct Flatten<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> Flatten<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Module<B> for Flatten<B>
where
    <Float as DataType>::Primitive: Element,
{
    type Input = Tensor<B, 2, Float>;
    type Output = Tensor<B, 2, Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        // Already 2D, so just pass through
        // In a full implementation, this would handle arbitrary dimensions
        input
    }
}

impl<B: Backend> Default for Flatten<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_linear_layer() {
        let layer = Linear::<CpuBackend>::new(3, 2);

        assert_eq!(layer.input_size(), 3);
        assert_eq!(layer.output_size(), 2);
        assert_eq!(layer.weights().shape().dims(), &[3, 2]);
        assert_eq!(layer.bias().shape().dims(), &[2]);

        // Test forward pass
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3]));
        let output = layer.forward(input);

        assert_eq!(output.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_relu_layer() {
        let layer = ReLU::<CpuBackend>::new();
        let input = Tensor::from_data(vec![-1.0, 0.0, 1.0, 2.0], Shape::new([2, 2]));
        let output = layer.forward(input);

        assert_eq!(output.to_data(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sigmoid_layer() {
        let layer = Sigmoid::<CpuBackend>::new();
        let input = Tensor::from_data(vec![0.0, 1.0], Shape::new([1, 2]));
        let output = layer.forward(input);
        let data = output.to_data();

        assert!((data[0] - 0.5).abs() < 1e-6);
        assert!(data[1] > 0.7 && data[1] < 0.8);
    }

    #[test]
    fn test_softmax_layer() {
        let layer = Softmax::<CpuBackend>::new();
        let input = Tensor::from_data(vec![1.0, 2.0, 3.0], Shape::new([1, 3]));
        let output = layer.forward(input);
        let data = output.to_data();

        // Check that probabilities sum to 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all values are positive
        assert!(data.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_leaky_relu_layer() {
        let layer = LeakyReLU::<CpuBackend>::new(0.1);
        let input = Tensor::from_data(vec![-1.0, 0.0, 1.0], Shape::new([1, 3]));
        let output = layer.forward(input);
        let data = output.to_data();

        assert_eq!(data[0], -0.1); // -1.0 * 0.1
        assert_eq!(data[1], 0.0); // 0.0
        assert_eq!(data[2], 1.0); // 1.0
    }

    #[test]
    fn test_layer_composition() {
        let linear = Linear::<CpuBackend>::new(2, 3);
        let relu = ReLU::<CpuBackend>::new();
        let softmax = Softmax::<CpuBackend>::new();

        let input = Tensor::from_data(vec![1.0, 2.0], Shape::new([1, 2]));

        // Chain the layers manually
        let x1 = linear.forward(input);
        let x2 = relu.forward(x1);
        let output = softmax.forward(x2);

        assert_eq!(output.shape().dims(), &[1, 3]);

        // Check softmax output
        let data = output.to_data();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_processing() {
        let layer = Linear::<CpuBackend>::new(2, 3);

        // Test with batch of 4 samples
        let input = Tensor::from_data(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            Shape::new([4, 2]),
        );

        let output = layer.forward(input);
        assert_eq!(output.shape().dims(), &[4, 3]);
    }
}
