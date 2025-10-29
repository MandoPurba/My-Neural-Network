//! Training Module
//!
//! This module provides training utilities for neural networks including
//! training loops, metrics tracking, and epoch management.

pub mod loop_trainer;
pub mod metrics;

use crate::backend::Backend;
use crate::optim::Optimizer;

pub use loop_trainer::{TrainingConfig, TrainingLoop, TrainingState};
pub use metrics::{Metrics, MetricsTracker, TrainingMetrics};

/// Result of a training step
#[derive(Debug, Clone)]
pub struct StepResult {
    /// Loss value for this step
    pub loss: f32,
    /// Number of samples processed
    pub samples: usize,
    /// Step number
    pub step: usize,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f32>,
}

impl StepResult {
    /// Create a new step result
    pub fn new(loss: f32, samples: usize, step: usize) -> Self {
        Self {
            loss,
            samples,
            step,
            metrics: std::collections::HashMap::new(),
        }
    }

    /// Add a metric to the step result
    pub fn with_metric(mut self, name: String, value: f32) -> Self {
        self.metrics.insert(name, value);
        self
    }

    /// Get a metric value
    pub fn get_metric(&self, name: &str) -> Option<f32> {
        self.metrics.get(name).copied()
    }
}

/// Result of a full epoch
#[derive(Debug, Clone)]
pub struct EpochResult {
    /// Epoch number
    pub epoch: usize,
    /// Average training loss
    pub train_loss: f32,
    /// Training accuracy (if applicable)
    pub train_accuracy: Option<f32>,
    /// Validation loss (if validation data provided)
    pub val_loss: Option<f32>,
    /// Validation accuracy (if validation data provided)
    pub val_accuracy: Option<f32>,
    /// Number of training samples
    pub train_samples: usize,
    /// Number of validation samples
    pub val_samples: usize,
    /// Time taken for this epoch in seconds
    pub duration: f32,
    /// Additional metrics
    pub metrics: std::collections::HashMap<String, f32>,
}

impl EpochResult {
    /// Create a new epoch result
    pub fn new(epoch: usize, train_loss: f32, train_samples: usize) -> Self {
        Self {
            epoch,
            train_loss,
            train_accuracy: None,
            val_loss: None,
            val_accuracy: None,
            train_samples,
            val_samples: 0,
            duration: 0.0,
            metrics: std::collections::HashMap::new(),
        }
    }

    /// Set validation results
    pub fn with_validation(mut self, val_loss: f32, val_samples: usize) -> Self {
        self.val_loss = Some(val_loss);
        self.val_samples = val_samples;
        self
    }

    /// Set training accuracy
    pub fn with_train_accuracy(mut self, accuracy: f32) -> Self {
        self.train_accuracy = Some(accuracy);
        self
    }

    /// Set validation accuracy
    pub fn with_val_accuracy(mut self, accuracy: f32) -> Self {
        self.val_accuracy = Some(accuracy);
        self
    }

    /// Set duration
    pub fn with_duration(mut self, duration: f32) -> Self {
        self.duration = duration;
        self
    }

    /// Add a metric
    pub fn with_metric(mut self, name: String, value: f32) -> Self {
        self.metrics.insert(name, value);
        self
    }
}

/// Training callback trait for custom behavior during training
pub trait TrainingCallback<B: Backend> {
    /// Called at the start of training
    fn on_training_start(&mut self, _config: &TrainingConfig) {}

    /// Called at the end of training
    fn on_training_end(&mut self, _results: &[EpochResult]) {}

    /// Called at the start of each epoch
    fn on_epoch_start(&mut self, _epoch: usize) {}

    /// Called at the end of each epoch
    fn on_epoch_end(&mut self, _result: &EpochResult) {}

    /// Called after each training step
    fn on_step_end(&mut self, _result: &StepResult) {}

    /// Called after validation (if applicable)
    fn on_validation_end(&mut self, _val_loss: f32, _val_accuracy: Option<f32>) {}

    /// Called when a new best model is found (based on validation loss)
    fn on_best_model(&mut self, _epoch: usize, _val_loss: f32) {}
}

/// Simple logging callback
pub struct LoggingCallback {
    /// Whether to log step-level information
    pub log_steps: bool,
    /// How often to log steps (every N steps)
    pub step_frequency: usize,
    /// How often to log epochs (every N epochs)
    pub epoch_frequency: usize,
}

impl LoggingCallback {
    /// Create a new logging callback
    pub fn new() -> Self {
        Self {
            log_steps: false,
            step_frequency: 100,
            epoch_frequency: 1,
        }
    }

    /// Enable step logging
    pub fn with_step_logging(mut self, frequency: usize) -> Self {
        self.log_steps = true;
        self.step_frequency = frequency;
        self
    }

    /// Set epoch logging frequency
    pub fn with_epoch_frequency(mut self, frequency: usize) -> Self {
        self.epoch_frequency = frequency;
        self
    }
}

impl<B: Backend> TrainingCallback<B> for LoggingCallback {
    fn on_training_start(&mut self, config: &TrainingConfig) {
        println!("Starting training with config:");
        println!("  Epochs: {}", config.epochs);
        println!("  Learning rate: {}", config.learning_rate);
        println!("  Batch size: {}", config.batch_size);
    }

    fn on_training_end(&mut self, results: &[EpochResult]) {
        println!("Training completed!");
        if let Some(last_result) = results.last() {
            println!("Final train loss: {:.4}", last_result.train_loss);
            if let Some(val_loss) = last_result.val_loss {
                println!("Final val loss: {:.4}", val_loss);
            }
        }
    }

    fn on_epoch_end(&mut self, result: &EpochResult) {
        if result.epoch % self.epoch_frequency == 0 {
            print!(
                "Epoch {}: train_loss={:.4}",
                result.epoch, result.train_loss
            );

            if let Some(train_acc) = result.train_accuracy {
                print!(", train_acc={:.4}", train_acc);
            }

            if let Some(val_loss) = result.val_loss {
                print!(", val_loss={:.4}", val_loss);
            }

            if let Some(val_acc) = result.val_accuracy {
                print!(", val_acc={:.4}", val_acc);
            }

            println!(" ({:.2}s)", result.duration);
        }
    }

    fn on_step_end(&mut self, result: &StepResult) {
        if self.log_steps && result.step % self.step_frequency == 0 {
            println!("Step {}: loss={:.4}", result.step, result.loss);
        }
    }

    fn on_best_model(&mut self, epoch: usize, val_loss: f32) {
        println!(
            "New best model at epoch {} with validation loss: {:.4}",
            epoch, val_loss
        );
    }
}

impl Default for LoggingCallback {
    fn default() -> Self {
        Self::new()
    }
}

/// Early stopping callback
pub struct EarlyStoppingCallback {
    /// Patience (number of epochs to wait for improvement)
    pub patience: usize,
    /// Minimum change to qualify as improvement
    pub min_delta: f32,
    /// Current best validation loss
    best_val_loss: Option<f32>,
    /// Epochs without improvement
    epochs_without_improvement: usize,
    /// Whether to stop training
    should_stop: bool,
}

impl EarlyStoppingCallback {
    /// Create a new early stopping callback
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best_val_loss: None,
            epochs_without_improvement: 0,
            should_stop: false,
        }
    }

    /// Check if training should stop
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    /// Reset the callback state
    pub fn reset(&mut self) {
        self.best_val_loss = None;
        self.epochs_without_improvement = 0;
        self.should_stop = false;
    }
}

impl<B: Backend> TrainingCallback<B> for EarlyStoppingCallback {
    fn on_training_start(&mut self, _config: &TrainingConfig) {
        self.reset();
    }

    fn on_validation_end(&mut self, val_loss: f32, _val_accuracy: Option<f32>) {
        match self.best_val_loss {
            None => {
                self.best_val_loss = Some(val_loss);
                self.epochs_without_improvement = 0;
            }
            Some(best) => {
                if val_loss < best - self.min_delta {
                    self.best_val_loss = Some(val_loss);
                    self.epochs_without_improvement = 0;
                } else {
                    self.epochs_without_improvement += 1;
                    if self.epochs_without_improvement >= self.patience {
                        self.should_stop = true;
                        println!(
                            "Early stopping triggered after {} epochs without improvement",
                            self.patience
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_step_result() {
        let mut result = StepResult::new(0.5, 32, 10);
        assert_eq!(result.loss, 0.5);
        assert_eq!(result.samples, 32);
        assert_eq!(result.step, 10);
        assert!(result.metrics.is_empty());

        result = result.with_metric("accuracy".to_string(), 0.85);
        assert_eq!(result.get_metric("accuracy"), Some(0.85));
        assert_eq!(result.get_metric("precision"), None);
    }

    #[test]
    fn test_epoch_result() {
        let mut result = EpochResult::new(1, 0.3, 1000);
        assert_eq!(result.epoch, 1);
        assert_eq!(result.train_loss, 0.3);
        assert_eq!(result.train_samples, 1000);
        assert_eq!(result.val_samples, 0);
        assert!(result.train_accuracy.is_none());
        assert!(result.val_loss.is_none());

        result = result
            .with_validation(0.25, 200)
            .with_train_accuracy(0.9)
            .with_val_accuracy(0.85)
            .with_duration(10.5)
            .with_metric("f1_score".to_string(), 0.87);

        assert_eq!(result.val_loss, Some(0.25));
        assert_eq!(result.val_samples, 200);
        assert_eq!(result.train_accuracy, Some(0.9));
        assert_eq!(result.val_accuracy, Some(0.85));
        assert_eq!(result.duration, 10.5);
        assert_eq!(result.metrics.get("f1_score"), Some(&0.87));
    }

    #[test]
    fn test_logging_callback() {
        let callback = LoggingCallback::new()
            .with_step_logging(50)
            .with_epoch_frequency(5);

        assert!(callback.log_steps);
        assert_eq!(callback.step_frequency, 50);
        assert_eq!(callback.epoch_frequency, 5);
    }

    #[test]
    fn test_early_stopping_callback() {
        let mut callback = EarlyStoppingCallback::new(3, 0.01);
        assert!(!callback.should_stop());

        // First validation - no previous best
        callback.on_validation_end(0.5, None);
        assert!(!callback.should_stop());
        assert_eq!(callback.best_val_loss, Some(0.5));

        // Second validation - improvement
        callback.on_validation_end(0.4, None);
        assert!(!callback.should_stop());
        assert_eq!(callback.best_val_loss, Some(0.4));
        assert_eq!(callback.epochs_without_improvement, 0);

        // Third validation - no improvement
        callback.on_validation_end(0.41, None);
        assert!(!callback.should_stop());
        assert_eq!(callback.epochs_without_improvement, 1);

        // Fourth validation - no improvement
        callback.on_validation_end(0.42, None);
        assert!(!callback.should_stop());
        assert_eq!(callback.epochs_without_improvement, 2);

        // Fifth validation - no improvement, should trigger early stopping
        callback.on_validation_end(0.43, None);
        assert!(callback.should_stop());
        assert_eq!(callback.epochs_without_improvement, 3);
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut callback = EarlyStoppingCallback::new(2, 0.01);
        callback.on_validation_end(0.5, None);
        callback.on_validation_end(0.6, None);
        callback.on_validation_end(0.7, None);
        assert!(callback.should_stop());

        callback.reset();
        assert!(!callback.should_stop());
        assert!(callback.best_val_loss.is_none());
        assert_eq!(callback.epochs_without_improvement, 0);
    }
}
