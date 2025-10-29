//! Training Loop Implementation
//!
//! This module provides the main training loop implementation with epoch management,
//! validation, and metrics tracking.

use crate::backend::Backend;
use crate::nn::loss::Loss;
use crate::optim::{Optimizer, Parameter};
use crate::train::{EpochResult, StepResult, TrainingCallback};
use std::time::Instant;

/// Configuration for training
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Number of epochs to train
    pub epochs: usize,
    /// Learning rate
    pub learning_rate: f32,
    /// Batch size
    pub batch_size: usize,
    /// Whether to shuffle data each epoch
    pub shuffle: bool,
    /// Whether to validate after each epoch
    pub validate: bool,
    /// How often to validate (every N epochs)
    pub validation_frequency: usize,
    /// Whether to save best model based on validation loss
    pub save_best: bool,
    /// Whether to compute accuracy metrics
    pub compute_accuracy: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl TrainingConfig {
    /// Create a new training configuration
    pub fn new(epochs: usize, learning_rate: f32, batch_size: usize) -> Self {
        Self {
            epochs,
            learning_rate,
            batch_size,
            shuffle: true,
            validate: true,
            validation_frequency: 1,
            save_best: false,
            compute_accuracy: false,
            seed: None,
        }
    }

    /// Disable validation
    pub fn without_validation(mut self) -> Self {
        self.validate = false;
        self
    }

    /// Set validation frequency
    pub fn with_validation_frequency(mut self, frequency: usize) -> Self {
        self.validation_frequency = frequency;
        self
    }

    /// Enable saving best model
    pub fn with_save_best(mut self) -> Self {
        self.save_best = true;
        self
    }

    /// Enable accuracy computation
    pub fn with_accuracy(mut self) -> Self {
        self.compute_accuracy = true;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Disable data shuffling
    pub fn without_shuffle(mut self) -> Self {
        self.shuffle = false;
        self
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new(10, 0.001, 32)
    }
}

/// Current state of training
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch
    pub current_epoch: usize,
    /// Current step (across all epochs)
    pub current_step: usize,
    /// Best validation loss seen so far
    pub best_val_loss: Option<f32>,
    /// Epoch with best validation loss
    pub best_epoch: Option<usize>,
    /// Whether training should stop (e.g., due to early stopping)
    pub should_stop: bool,
    /// Training results for each completed epoch
    pub epoch_results: Vec<EpochResult>,
}

impl TrainingState {
    /// Create a new training state
    pub fn new() -> Self {
        Self {
            current_epoch: 0,
            current_step: 0,
            best_val_loss: None,
            best_epoch: None,
            should_stop: false,
            epoch_results: Vec::new(),
        }
    }

    /// Check if this is the best model so far
    pub fn is_best_model(&self, val_loss: f32) -> bool {
        match self.best_val_loss {
            None => true,
            Some(best) => val_loss < best,
        }
    }

    /// Update best model information
    pub fn update_best_model(&mut self, val_loss: f32, epoch: usize) {
        self.best_val_loss = Some(val_loss);
        self.best_epoch = Some(epoch);
    }

    /// Get the number of completed epochs
    pub fn completed_epochs(&self) -> usize {
        self.epoch_results.len()
    }

    /// Get the last epoch result
    pub fn last_epoch_result(&self) -> Option<&EpochResult> {
        self.epoch_results.last()
    }
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

/// Main training loop implementation
pub struct TrainingLoop<B: Backend> {
    /// Training configuration
    config: TrainingConfig,
    /// Current training state
    state: TrainingState,
    /// Training callbacks
    callbacks: Vec<Box<dyn TrainingCallback<B>>>,
}

impl<B: Backend + Default> TrainingLoop<B> {
    /// Create a new training loop
    pub fn new(config: TrainingConfig) -> Self {
        Self {
            config,
            state: TrainingState::new(),
            callbacks: Vec::new(),
        }
    }

    /// Add a callback to the training loop
    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback<B>>) {
        self.callbacks.push(callback);
    }

    /// Get the current training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }

    /// Get the training configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }

    /// Train a model with the given data
    pub fn train<M, L, O, D>(
        &mut self,
        model: &mut M,
        loss_fn: &L,
        optimizer: &mut O,
        train_data: &D,
        val_data: Option<&D>,
    ) -> Result<Vec<EpochResult>, TrainingError>
    where
        M: Model<B>,
        L: Loss<B>,
        O: Optimizer<B>,
        D: Dataset<B>,
    {
        // Initialize training
        self.on_training_start();
        optimizer.set_learning_rate(self.config.learning_rate);

        // Main training loop
        for epoch in 1..=self.config.epochs {
            if self.state.should_stop {
                break;
            }

            self.state.current_epoch = epoch;
            self.on_epoch_start(epoch);

            let epoch_start = Instant::now();

            // Training phase
            let train_result = self.train_epoch(model, loss_fn, optimizer, train_data)?;

            // Validation phase
            let val_result = if self.config.validate
                && epoch % self.config.validation_frequency == 0
                && val_data.is_some()
            {
                Some(self.validate_epoch(model, loss_fn, val_data.unwrap())?)
            } else {
                None
            };

            let epoch_duration = epoch_start.elapsed().as_secs_f32();

            // Create epoch result
            let mut epoch_result = EpochResult::new(epoch, train_result.0, train_result.1)
                .with_duration(epoch_duration);

            if let Some(train_acc) = train_result.2 {
                epoch_result = epoch_result.with_train_accuracy(train_acc);
            }

            if let Some((val_loss, val_samples, val_acc)) = val_result {
                epoch_result = epoch_result.with_validation(val_loss, val_samples);
                if let Some(acc) = val_acc {
                    epoch_result = epoch_result.with_val_accuracy(acc);
                }

                // Check for best model
                if self.state.is_best_model(val_loss) {
                    self.state.update_best_model(val_loss, epoch);
                    self.on_best_model(epoch, val_loss);
                }

                self.on_validation_end(val_loss, val_acc);
            }

            self.state.epoch_results.push(epoch_result.clone());
            self.on_epoch_end(&epoch_result);
        }

        self.on_training_end();
        Ok(self.state.epoch_results.clone())
    }

    /// Train for one epoch
    fn train_epoch<M, L, O, D>(
        &mut self,
        model: &mut M,
        loss_fn: &L,
        optimizer: &mut O,
        data: &D,
    ) -> Result<(f32, usize, Option<f32>), TrainingError>
    where
        M: Model<B>,
        L: Loss<B>,
        O: Optimizer<B>,
        D: Dataset<B>,
    {
        model.train();
        let mut total_loss = 0.0;
        let mut total_samples = 0;
        let mut correct_predictions = 0;

        for (batch_idx, batch) in data.batches(self.config.batch_size).enumerate() {
            let (inputs, targets) = batch?;
            let batch_size = inputs.len();

            // Zero gradients
            let mut parameters = model.parameters();
            optimizer.zero_grad(&mut parameters);

            // Forward pass
            let outputs = model.forward(&inputs)?;

            // Compute loss
            let loss = loss_fn.forward(&outputs, &targets)?;
            total_loss += loss * batch_size as f32;
            total_samples += batch_size;

            // Compute accuracy if requested
            if self.config.compute_accuracy {
                let predictions = self.get_predictions(&outputs)?;
                correct_predictions += self.count_correct(&predictions, &targets);
            }

            // Backward pass
            // In a full implementation, this would call loss.backward() and propagate gradients
            // For now, this is a placeholder

            // Update parameters
            optimizer.step(&mut parameters);

            // Create step result
            let step_result = StepResult::new(loss, batch_size, self.state.current_step);
            self.state.current_step += 1;
            self.on_step_end(&step_result);
        }

        let avg_loss = total_loss / total_samples as f32;
        let accuracy = if self.config.compute_accuracy && total_samples > 0 {
            Some(correct_predictions as f32 / total_samples as f32)
        } else {
            None
        };

        Ok((avg_loss, total_samples, accuracy))
    }

    /// Validate for one epoch
    fn validate_epoch<M, L, D>(
        &mut self,
        model: &mut M,
        loss_fn: &L,
        data: &D,
    ) -> Result<(f32, usize, Option<f32>), TrainingError>
    where
        M: Model<B>,
        L: Loss<B>,
        D: Dataset<B>,
    {
        model.eval();
        let mut total_loss = 0.0;
        let mut total_samples = 0;
        let mut correct_predictions = 0;

        for batch in data.batches(self.config.batch_size) {
            let (inputs, targets) = batch?;
            let batch_size = inputs.len();

            // Forward pass (no gradients)
            let outputs = model.forward(&inputs)?;

            // Compute loss
            let loss = loss_fn.forward(&outputs, &targets)?;
            total_loss += loss * batch_size as f32;
            total_samples += batch_size;

            // Compute accuracy if requested
            if self.config.compute_accuracy {
                let predictions = self.get_predictions(&outputs)?;
                correct_predictions += self.count_correct(&predictions, &targets);
            }
        }

        let avg_loss = total_loss / total_samples as f32;
        let accuracy = if self.config.compute_accuracy && total_samples > 0 {
            Some(correct_predictions as f32 / total_samples as f32)
        } else {
            None
        };

        Ok((avg_loss, total_samples, accuracy))
    }

    /// Get predictions from model outputs
    fn get_predictions<T>(&self, outputs: &[T]) -> Result<Vec<usize>, TrainingError> {
        // This is a placeholder - in a real implementation, this would
        // convert model outputs to class predictions (e.g., argmax for classification)
        Ok(vec![0; outputs.len()])
    }

    /// Count correct predictions
    fn count_correct<T, U>(&self, predictions: &[T], targets: &[U]) -> usize {
        // This is a placeholder - in a real implementation, this would
        // compare predictions with ground truth targets
        0
    }

    // Callback methods
    fn on_training_start(&mut self) {
        for callback in &mut self.callbacks {
            callback.on_training_start(&self.config);
        }
    }

    fn on_training_end(&mut self) {
        for callback in &mut self.callbacks {
            callback.on_training_end(&self.state.epoch_results);
        }
    }

    fn on_epoch_start(&mut self, epoch: usize) {
        for callback in &mut self.callbacks {
            callback.on_epoch_start(epoch);
        }
    }

    fn on_epoch_end(&mut self, result: &EpochResult) {
        for callback in &mut self.callbacks {
            callback.on_epoch_end(result);
        }
    }

    fn on_step_end(&mut self, result: &StepResult) {
        for callback in &mut self.callbacks {
            callback.on_step_end(result);
        }
    }

    fn on_validation_end(&mut self, val_loss: f32, val_accuracy: Option<f32>) {
        for callback in &mut self.callbacks {
            callback.on_validation_end(val_loss, val_accuracy);
        }
    }

    fn on_best_model(&mut self, epoch: usize, val_loss: f32) {
        for callback in &mut self.callbacks {
            callback.on_best_model(epoch, val_loss);
        }
    }
}

/// Error types for training
#[derive(Debug)]
pub enum TrainingError {
    /// Model error
    ModelError(String),
    /// Data error
    DataError(String),
    /// Optimizer error
    OptimizerError(String),
    /// Loss function error
    LossError(String),
    /// General error
    General(String),
}

impl std::fmt::Display for TrainingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TrainingError::ModelError(msg) => write!(f, "Model error: {}", msg),
            TrainingError::DataError(msg) => write!(f, "Data error: {}", msg),
            TrainingError::OptimizerError(msg) => write!(f, "Optimizer error: {}", msg),
            TrainingError::LossError(msg) => write!(f, "Loss error: {}", msg),
            TrainingError::General(msg) => write!(f, "Training error: {}", msg),
        }
    }
}

impl std::error::Error for TrainingError {}

/// Trait for models that can be trained
pub trait Model<B: Backend> {
    /// Set the model to training mode
    fn train(&mut self);

    /// Set the model to evaluation mode
    fn eval(&mut self);

    /// Forward pass
    fn forward<T>(&mut self, inputs: &[T]) -> Result<Vec<T>, TrainingError>;

    /// Get model parameters
    fn parameters(&mut self) -> Vec<&mut dyn Parameter<B>>;
}

/// Trait for datasets
pub trait Dataset<B: Backend> {
    type Batch;
    type BatchIter: Iterator<Item = Result<Self::Batch, TrainingError>>;

    /// Get an iterator over batches
    fn batches(&self, batch_size: usize) -> Self::BatchIter;

    /// Get the total number of samples
    fn len(&self) -> usize;

    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::new(50, 0.001, 64);
        assert_eq!(config.epochs, 50);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 64);
        assert!(config.shuffle);
        assert!(config.validate);

        let config = config
            .without_validation()
            .with_validation_frequency(5)
            .with_save_best()
            .with_accuracy()
            .with_seed(42)
            .without_shuffle();

        assert!(!config.validate);
        assert_eq!(config.validation_frequency, 5);
        assert!(config.save_best);
        assert!(config.compute_accuracy);
        assert_eq!(config.seed, Some(42));
        assert!(!config.shuffle);
    }

    #[test]
    fn test_training_state() {
        let mut state = TrainingState::new();
        assert_eq!(state.current_epoch, 0);
        assert_eq!(state.current_step, 0);
        assert!(state.best_val_loss.is_none());
        assert!(!state.should_stop);

        // Test best model tracking
        assert!(state.is_best_model(0.5));
        state.update_best_model(0.5, 1);
        assert_eq!(state.best_val_loss, Some(0.5));
        assert_eq!(state.best_epoch, Some(1));

        assert!(state.is_best_model(0.4));
        assert!(!state.is_best_model(0.6));
    }

    #[test]
    fn test_default_config() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 10);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 32);
    }

    #[test]
    fn test_training_error_display() {
        let error = TrainingError::ModelError("Test error".to_string());
        assert_eq!(format!("{}", error), "Model error: Test error");

        let error = TrainingError::DataError("Data issue".to_string());
        assert_eq!(format!("{}", error), "Data error: Data issue");
    }
}
