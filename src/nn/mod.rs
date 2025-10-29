//! Neural Network Module
//!
//! This module provides high-level APIs for building neural networks with Mini-Burn.
//! It includes common layers, activation functions, loss functions, and model utilities.

use crate::backend::Backend;
use crate::data::Float;
use crate::shape::Shape;
use crate::tensor::Tensor;

pub mod init;
pub mod layers;
pub mod loss;

pub use init::*;
pub use layers::*;
pub use loss::*;

/// Trait for neural network modules/layers
pub trait Module<B: Backend> {
    type Input;
    type Output;

    /// Forward pass through the module
    fn forward(&self, input: Self::Input) -> Self::Output;
}

/// Parameter trait for trainable parameters
pub trait Parameter<B: Backend> {
    /// Get the parameter tensor
    fn param(&self) -> &Tensor<B, 2, Float>;

    /// Get mutable reference to parameter tensor (for updates)
    fn param_mut(&mut self) -> &mut Tensor<B, 2, Float>;
}

/// Sequential model container
pub struct Sequential<B: Backend> {
    layers: Vec<Box<dyn Module<B, Input = Tensor<B, 2, Float>, Output = Tensor<B, 2, Float>>>>,
}

impl<B: Backend + Default> Sequential<B> {
    /// Create new sequential model
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Add a layer to the model
    pub fn add<L>(mut self, layer: L) -> Self
    where
        L: Module<B, Input = Tensor<B, 2, Float>, Output = Tensor<B, 2, Float>> + 'static,
    {
        self.layers.push(Box::new(layer));
        self
    }

    /// Forward pass through all layers
    pub fn forward(&self, mut input: Tensor<B, 2, Float>) -> Tensor<B, 2, Float> {
        for layer in &self.layers {
            input = layer.forward(input);
        }
        input
    }

    /// Predict single sample (convert 1D to 2D internally)
    pub fn predict(&self, input: &Tensor<B, 1, Float>) -> Tensor<B, 1, Float> {
        // Convert 1D to 2D batch format
        let batch_input = Tensor::from_data(input.to_data(), Shape::new([1, input.shape().dim(0)]));

        // Forward pass
        let batch_output = self.forward(batch_input);

        // Convert back to 1D
        Tensor::from_data(
            batch_output.to_data(),
            Shape::new([batch_output.shape().dim(1)]),
        )
    }

    /// Get number of layers
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if model is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }
}

impl<B: Backend + Default> Default for Sequential<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Model building utilities
pub struct ModelBuilder<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend + Default> ModelBuilder<B> {
    /// Create a simple fully connected classifier
    pub fn classifier(
        input_size: usize,
        hidden_sizes: &[usize],
        num_classes: usize,
    ) -> Sequential<B> {
        let mut model = Sequential::new();

        let mut prev_size = input_size;

        // Add hidden layers
        for &hidden_size in hidden_sizes {
            model = model
                .add(Linear::new(prev_size, hidden_size))
                .add(ReLU::new());
            prev_size = hidden_size;
        }

        // Add output layer with softmax
        model = model
            .add(Linear::new(prev_size, num_classes))
            .add(Softmax::new());

        model
    }

    /// Create a simple multilayer perceptron
    pub fn mlp(layer_sizes: &[usize]) -> Sequential<B> {
        let mut model = Sequential::new();

        for window in layer_sizes.windows(2) {
            let input_size = window[0];
            let output_size = window[1];

            model = model.add(Linear::new(input_size, output_size));

            // Add ReLU for all layers except the last one
            if window != &layer_sizes[layer_sizes.len() - 2..] {
                model = model.add(ReLU::new());
            }
        }

        // Add softmax to output layer
        model = model.add(Softmax::new());

        model
    }
}

/// Training utilities (simplified without autodiff)
pub struct Trainer<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend + Default> Trainer<B> {
    /// Evaluate model on dataset
    pub fn evaluate(
        model: &Sequential<B>,
        images: &[Tensor<B, 1, Float>],
        labels: &[usize],
    ) -> (f32, Vec<f32>) {
        let mut correct = 0;
        let total = images.len();
        let num_classes = labels.iter().max().unwrap_or(&0) + 1;
        let mut class_correct = vec![0; num_classes];
        let mut class_total = vec![0; num_classes];

        for (image, &label) in images.iter().zip(labels.iter()) {
            let output = model.predict(image);
            let probs = output.to_data();

            let predicted = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            class_total[label] += 1;
            if predicted == label {
                correct += 1;
                class_correct[label] += 1;
            }
        }

        let overall_accuracy = correct as f32 / total as f32;
        let class_accuracies: Vec<f32> = class_correct
            .iter()
            .zip(class_total.iter())
            .map(|(&correct, &total)| {
                if total > 0 {
                    correct as f32 / total as f32
                } else {
                    0.0
                }
            })
            .collect();

        (overall_accuracy, class_accuracies)
    }

    /// Simple batch inference
    pub fn batch_predict(model: &Sequential<B>, batch: &Tensor<B, 2, Float>) -> Vec<(usize, f32)> {
        let output = model.forward(batch.clone());
        let batch_size = batch.shape().dim(0);
        let num_classes = output.shape().dim(1);
        let probs = output.to_data();

        let mut predictions = Vec::new();
        for i in 0..batch_size {
            let start_idx = i * num_classes;
            let sample_probs = &probs[start_idx..start_idx + num_classes];

            let (predicted_class, &confidence) = sample_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            predictions.push((predicted_class, confidence));
        }

        predictions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_sequential_model() {
        let model = Sequential::<CpuBackend>::new()
            .add(Linear::new(10, 5))
            .add(ReLU::new())
            .add(Linear::new(5, 3))
            .add(Softmax::new());

        assert_eq!(model.len(), 4);

        // Test forward pass
        let input = Tensor::from_data(vec![1.0; 10], Shape::new([10]));
        let output = model.predict(&input);

        assert_eq!(output.shape().dims(), &[3]);

        // Check that softmax output sums to ~1
        let sum: f32 = output.to_data().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_model_builder() {
        let model = ModelBuilder::<CpuBackend>::classifier(784, &[128, 64], 10);
        assert_eq!(model.len(), 6); // Linear + ReLU + Linear + ReLU + Linear + Softmax

        let mlp = ModelBuilder::<CpuBackend>::mlp(&[784, 128, 64, 10]);
        assert_eq!(mlp.len(), 6); // 3 Linear + 2 ReLU + Softmax
    }

    #[test]
    fn test_trainer_evaluation() {
        let model = ModelBuilder::<CpuBackend>::classifier(4, &[8], 2);

        // Create dummy data
        let images = vec![
            Tensor::from_data(vec![1.0, 0.0, 0.0, 0.0], Shape::new([4])),
            Tensor::from_data(vec![0.0, 1.0, 0.0, 0.0], Shape::new([4])),
        ];
        let labels = vec![0, 1];

        let (accuracy, class_accuracies) = Trainer::evaluate(&model, &images, &labels);

        assert!(accuracy >= 0.0 && accuracy <= 1.0);
        assert_eq!(class_accuracies.len(), 2);
    }
}
