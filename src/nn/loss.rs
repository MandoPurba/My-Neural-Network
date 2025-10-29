//! Loss Functions
//!
//! This module provides common loss functions used in neural network training
//! and evaluation. While our framework doesn't support automatic differentiation
//! yet, these functions are useful for model evaluation and understanding.

use crate::backend::Backend;
use crate::data::{DataType, Element, Float};
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Trait for loss functions
pub trait Loss<B: Backend + Default> {
    /// Compute loss between predictions and targets
    fn compute(&self, predictions: &Tensor<B, 2, Float>, targets: &Tensor<B, 2, Float>) -> f32;

    /// Compute loss for single prediction
    fn compute_single(&self, prediction: &Tensor<B, 1, Float>, target: &Tensor<B, 1, Float>) -> f32
    where
        <Float as DataType>::Primitive: Element,
    {
        // Convert to batch format and compute
        let pred_batch = Tensor::from_data(
            prediction.to_data(),
            Shape::new([1, prediction.shape().dim(0)]),
        );
        let target_batch =
            Tensor::from_data(target.to_data(), Shape::new([1, target.shape().dim(0)]));
        self.compute(&pred_batch, &target_batch)
    }
}

/// Mean Squared Error Loss
pub struct MSELoss<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> MSELoss<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Loss<B> for MSELoss<B>
where
    <Float as DataType>::Primitive: Element,
{
    fn compute(&self, predictions: &Tensor<B, 2, Float>, targets: &Tensor<B, 2, Float>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have same shape"
        );

        let pred_data = predictions.to_data();
        let target_data = targets.to_data();

        let mse: f32 = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f32>()
            / pred_data.len() as f32;

        mse
    }
}

impl<B: Backend> Default for MSELoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Cross Entropy Loss for classification
pub struct CrossEntropyLoss<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> CrossEntropyLoss<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> CrossEntropyLoss<B>
where
    <Float as DataType>::Primitive: Element,
{
    /// Compute cross entropy loss with class indices
    pub fn compute_with_indices(
        &self,
        predictions: &Tensor<B, 2, Float>,
        target_indices: &[usize],
    ) -> f32 {
        let pred_data = predictions.to_data();
        let batch_size = predictions.shape().dim(0);
        let num_classes = predictions.shape().dim(1);

        assert_eq!(
            batch_size,
            target_indices.len(),
            "Batch size must match number of target indices"
        );

        let mut total_loss = 0.0;
        for (i, &target_idx) in target_indices.iter().enumerate() {
            assert!(target_idx < num_classes, "Target index out of bounds");

            // Get prediction for the target class
            let pred_prob = pred_data[i * num_classes + target_idx];

            // Avoid log(0) by adding small epsilon
            let eps = 1e-7;
            let clamped_prob = pred_prob.max(eps);

            total_loss += -clamped_prob.ln();
        }

        total_loss / batch_size as f32
    }
}

impl<B: Backend + Default> Loss<B> for CrossEntropyLoss<B>
where
    <Float as DataType>::Primitive: Element,
{
    fn compute(&self, predictions: &Tensor<B, 2, Float>, targets: &Tensor<B, 2, Float>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have same shape"
        );

        let pred_data = predictions.to_data();
        let target_data = targets.to_data();
        let batch_size = predictions.shape().dim(0);
        let num_classes = predictions.shape().dim(1);

        let mut total_loss = 0.0;
        for i in 0..batch_size {
            let mut sample_loss = 0.0;
            for j in 0..num_classes {
                let idx = i * num_classes + j;
                let pred_prob = pred_data[idx];
                let target_prob = target_data[idx];

                if target_prob > 0.0 {
                    // Avoid log(0) by adding small epsilon
                    let eps = 1e-7;
                    let clamped_prob = pred_prob.max(eps);
                    sample_loss += target_prob * (-clamped_prob.ln());
                }
            }
            total_loss += sample_loss;
        }

        total_loss / batch_size as f32
    }
}

impl<B: Backend> Default for CrossEntropyLoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Binary Cross Entropy Loss
pub struct BCELoss<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> BCELoss<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Loss<B> for BCELoss<B>
where
    <Float as DataType>::Primitive: Element,
{
    fn compute(&self, predictions: &Tensor<B, 2, Float>, targets: &Tensor<B, 2, Float>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have same shape"
        );

        let pred_data = predictions.to_data();
        let target_data = targets.to_data();

        let eps = 1e-7;
        let bce: f32 = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(p, t)| {
                let clamped_p = p.max(eps).min(1.0 - eps);
                -(t * clamped_p.ln() + (1.0 - t) * (1.0 - clamped_p).ln())
            })
            .sum::<f32>()
            / pred_data.len() as f32;

        bce
    }
}

impl<B: Backend> Default for BCELoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Mean Absolute Error Loss (L1 Loss)
pub struct MAELoss<B: Backend> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> MAELoss<B> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default> Loss<B> for MAELoss<B>
where
    <Float as DataType>::Primitive: Element,
{
    fn compute(&self, predictions: &Tensor<B, 2, Float>, targets: &Tensor<B, 2, Float>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have same shape"
        );

        let pred_data = predictions.to_data();
        let target_data = targets.to_data();

        let mae: f32 = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(p, t)| (p - t).abs())
            .sum::<f32>()
            / pred_data.len() as f32;

        mae
    }
}

impl<B: Backend> Default for MAELoss<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// Huber Loss (combines MSE and MAE)
pub struct HuberLoss<B: Backend> {
    delta: f32,
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend> HuberLoss<B> {
    pub fn new(delta: f32) -> Self {
        Self {
            delta,
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn with_default_delta() -> Self {
        Self::new(1.0)
    }
}

impl<B: Backend + Default> Loss<B> for HuberLoss<B>
where
    <Float as DataType>::Primitive: Element,
{
    fn compute(&self, predictions: &Tensor<B, 2, Float>, targets: &Tensor<B, 2, Float>) -> f32 {
        assert_eq!(
            predictions.shape(),
            targets.shape(),
            "Predictions and targets must have same shape"
        );

        let pred_data = predictions.to_data();
        let target_data = targets.to_data();

        let huber: f32 = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(p, t)| {
                let diff = (p - t).abs();
                if diff <= self.delta {
                    0.5 * diff.powi(2)
                } else {
                    self.delta * diff - 0.5 * self.delta.powi(2)
                }
            })
            .sum::<f32>()
            / pred_data.len() as f32;

        huber
    }
}

/// Utility functions for creating one-hot encodings
pub struct LossUtils;

impl LossUtils {
    /// Convert class indices to one-hot encoding
    pub fn indices_to_one_hot<B: Backend + Default>(
        indices: &[usize],
        num_classes: usize,
    ) -> Tensor<B, 2, Float> {
        let batch_size = indices.len();
        let mut data = vec![0.0; batch_size * num_classes];

        for (i, &class_idx) in indices.iter().enumerate() {
            if class_idx < num_classes {
                data[i * num_classes + class_idx] = 1.0;
            }
        }

        Tensor::from_data(data, Shape::new([batch_size, num_classes]))
    }

    /// Convert predictions to class indices (argmax)
    pub fn predictions_to_indices<B: Backend + Default>(
        predictions: &Tensor<B, 2, Float>,
    ) -> Vec<usize> {
        let pred_data = predictions.to_data();
        let batch_size = predictions.shape().dim(0);
        let num_classes = predictions.shape().dim(1);

        let mut indices = Vec::new();
        for i in 0..batch_size {
            let start_idx = i * num_classes;
            let sample_probs = &pred_data[start_idx..start_idx + num_classes];

            let predicted_class = sample_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            indices.push(predicted_class);
        }

        indices
    }

    /// Compute accuracy from predictions and target indices
    pub fn accuracy<B: Backend + Default>(
        predictions: &Tensor<B, 2, Float>,
        target_indices: &[usize],
    ) -> f32 {
        let predicted_indices = Self::predictions_to_indices(predictions);

        assert_eq!(
            predicted_indices.len(),
            target_indices.len(),
            "Number of predictions must match number of targets"
        );

        let correct = predicted_indices
            .iter()
            .zip(target_indices.iter())
            .filter(|(pred, target)| pred == target)
            .count();

        correct as f32 / predicted_indices.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_mse_loss() {
        let loss = MSELoss::<CpuBackend>::new();

        let predictions = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
        let targets = Tensor::from_data(vec![1.5, 2.5, 2.5, 3.5], Shape::new([2, 2]));

        let mse = loss.compute(&predictions, &targets);
        assert!((mse - 0.25).abs() < 1e-6); // Expected: (0.5² + 0.5² + 0.5² + 0.5²) / 4 = 0.25
    }

    #[test]
    fn test_cross_entropy_with_indices() {
        let loss = CrossEntropyLoss::<CpuBackend>::new();

        // Perfect predictions (probability 1.0 for correct class)
        let predictions = Tensor::from_data(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], Shape::new([2, 3]));
        let target_indices = vec![0, 1];

        let ce = loss.compute_with_indices(&predictions, &target_indices);
        assert!(ce < 1e-6); // Should be close to 0 for perfect predictions
    }

    #[test]
    fn test_bce_loss() {
        let loss = BCELoss::<CpuBackend>::new();

        let predictions = Tensor::from_data(vec![0.8, 0.2, 0.9, 0.1], Shape::new([2, 2]));
        let targets = Tensor::from_data(vec![1.0, 0.0, 1.0, 0.0], Shape::new([2, 2]));

        let bce = loss.compute(&predictions, &targets);
        assert!(bce > 0.0); // Should be positive
    }

    #[test]
    fn test_mae_loss() {
        let loss = MAELoss::<CpuBackend>::new();

        let predictions = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
        let targets = Tensor::from_data(vec![1.5, 2.5, 2.5, 3.5], Shape::new([2, 2]));

        let mae = loss.compute(&predictions, &targets);
        assert!((mae - 0.5).abs() < 1e-6); // Expected: (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
    }

    #[test]
    fn test_huber_loss() {
        let loss = HuberLoss::<CpuBackend>::new(1.0);

        let predictions = Tensor::from_data(vec![1.0, 2.0], Shape::new([1, 2]));
        let targets = Tensor::from_data(vec![1.5, 4.0], Shape::new([1, 2]));

        let huber = loss.compute(&predictions, &targets);
        // First: |1.0 - 1.5| = 0.5 <= 1.0, so 0.5 * 0.5² = 0.125
        // Second: |2.0 - 4.0| = 2.0 > 1.0, so 1.0 * 2.0 - 0.5 * 1.0² = 1.5
        // Average: (0.125 + 1.5) / 2 = 0.8125
        assert!((huber - 0.8125).abs() < 1e-6);
    }

    #[test]
    fn test_one_hot_conversion() {
        let indices = vec![0, 2, 1];
        let one_hot: Tensor<CpuBackend, 2> = LossUtils::indices_to_one_hot(&indices, 3);

        let expected = vec![
            1.0, 0.0, 0.0, // Class 0
            0.0, 0.0, 1.0, // Class 2
            0.0, 1.0, 0.0, // Class 1
        ];

        assert_eq!(one_hot.to_data(), expected);
    }

    #[test]
    fn test_predictions_to_indices() {
        let predictions: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![0.1, 0.9, 0.0, 0.0, 0.8, 0.1], Shape::new([2, 3]));

        let indices = LossUtils::predictions_to_indices(&predictions);
        assert_eq!(indices, vec![1, 1]); // Max indices for each sample
    }

    #[test]
    fn test_accuracy() {
        let predictions: Tensor<CpuBackend, 2> = Tensor::from_data(
            vec![0.1, 0.9, 0.0, 0.0, 0.8, 0.1, 0.2, 0.2, 0.6],
            Shape::new([3, 3]),
        );
        let targets = vec![1, 1, 2]; // Expected classes

        let acc = LossUtils::accuracy(&predictions, &targets);
        assert!((acc - 1.0).abs() < 1e-6); // All predictions should be correct
    }

    #[test]
    fn test_loss_single_sample() {
        let loss = MSELoss::<CpuBackend>::new();

        let prediction = Tensor::from_data(vec![1.0, 2.0], Shape::new([2]));
        let target = Tensor::from_data(vec![1.5, 2.5], Shape::new([2]));

        let mse = loss.compute_single(&prediction, &target);
        assert!((mse - 0.25).abs() < 1e-6);
    }
}
