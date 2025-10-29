//! Metrics Tracking Module
//!
//! This module provides utilities for tracking and computing various metrics
//! during training such as accuracy, precision, recall, F1-score, etc.

use std::collections::HashMap;

/// Trait for computing metrics
pub trait Metric {
    /// The type of predictions this metric expects
    type Prediction;
    /// The type of targets this metric expects
    type Target;

    /// Update the metric with new predictions and targets
    fn update(&mut self, predictions: &[Self::Prediction], targets: &[Self::Target]);

    /// Compute the current metric value
    fn compute(&self) -> f32;

    /// Reset the metric state
    fn reset(&mut self);

    /// Get the name of this metric
    fn name(&self) -> &str;
}

/// Accuracy metric for classification tasks
#[derive(Debug, Clone)]
pub struct Accuracy {
    /// Number of correct predictions
    correct: usize,
    /// Total number of predictions
    total: usize,
}

impl Accuracy {
    /// Create a new accuracy metric
    pub fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
        }
    }

    /// Get the number of correct predictions
    pub fn correct(&self) -> usize {
        self.correct
    }

    /// Get the total number of predictions
    pub fn total(&self) -> usize {
        self.total
    }
}

impl Default for Accuracy {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for Accuracy {
    type Prediction = usize;
    type Target = usize;

    fn update(&mut self, predictions: &[Self::Prediction], targets: &[Self::Target]) {
        assert_eq!(predictions.len(), targets.len());

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if pred == target {
                self.correct += 1;
            }
            self.total += 1;
        }
    }

    fn compute(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f32 / self.total as f32
        }
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn name(&self) -> &str {
        "accuracy"
    }
}

/// Precision metric for classification tasks
#[derive(Debug, Clone)]
pub struct Precision {
    /// True positives for each class
    true_positives: HashMap<usize, usize>,
    /// False positives for each class
    false_positives: HashMap<usize, usize>,
    /// Number of classes
    num_classes: usize,
    /// Whether to compute macro average
    macro_average: bool,
}

impl Precision {
    /// Create a new precision metric
    pub fn new(num_classes: usize, macro_average: bool) -> Self {
        Self {
            true_positives: HashMap::new(),
            false_positives: HashMap::new(),
            num_classes,
            macro_average,
        }
    }

    /// Create a binary precision metric
    pub fn binary() -> Self {
        Self::new(2, false)
    }

    /// Create a macro-averaged precision metric
    pub fn macro_avg(num_classes: usize) -> Self {
        Self::new(num_classes, true)
    }
}

impl Metric for Precision {
    type Prediction = usize;
    type Target = usize;

    fn update(&mut self, predictions: &[Self::Prediction], targets: &[Self::Target]) {
        assert_eq!(predictions.len(), targets.len());

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if pred == target {
                // True positive for this class
                *self.true_positives.entry(*pred).or_insert(0) += 1;
            } else {
                // False positive for predicted class
                *self.false_positives.entry(*pred).or_insert(0) += 1;
            }
        }
    }

    fn compute(&self) -> f32 {
        if self.macro_average {
            let mut total_precision = 0.0;
            let mut valid_classes = 0;

            for class in 0..self.num_classes {
                let tp = self.true_positives.get(&class).unwrap_or(&0);
                let fp = self.false_positives.get(&class).unwrap_or(&0);

                if tp + fp > 0 {
                    total_precision += *tp as f32 / (tp + fp) as f32;
                    valid_classes += 1;
                }
            }

            if valid_classes > 0 {
                total_precision / valid_classes as f32
            } else {
                0.0
            }
        } else {
            // Micro average (binary or overall)
            let total_tp: usize = self.true_positives.values().sum();
            let total_fp: usize = self.false_positives.values().sum();

            if total_tp + total_fp == 0 {
                0.0
            } else {
                total_tp as f32 / (total_tp + total_fp) as f32
            }
        }
    }

    fn reset(&mut self) {
        self.true_positives.clear();
        self.false_positives.clear();
    }

    fn name(&self) -> &str {
        if self.macro_average {
            "precision_macro"
        } else {
            "precision"
        }
    }
}

/// Recall metric for classification tasks
#[derive(Debug, Clone)]
pub struct Recall {
    /// True positives for each class
    true_positives: HashMap<usize, usize>,
    /// False negatives for each class
    false_negatives: HashMap<usize, usize>,
    /// Number of classes
    num_classes: usize,
    /// Whether to compute macro average
    macro_average: bool,
}

impl Recall {
    /// Create a new recall metric
    pub fn new(num_classes: usize, macro_average: bool) -> Self {
        Self {
            true_positives: HashMap::new(),
            false_negatives: HashMap::new(),
            num_classes,
            macro_average,
        }
    }

    /// Create a binary recall metric
    pub fn binary() -> Self {
        Self::new(2, false)
    }

    /// Create a macro-averaged recall metric
    pub fn macro_avg(num_classes: usize) -> Self {
        Self::new(num_classes, true)
    }
}

impl Metric for Recall {
    type Prediction = usize;
    type Target = usize;

    fn update(&mut self, predictions: &[Self::Prediction], targets: &[Self::Target]) {
        assert_eq!(predictions.len(), targets.len());

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            if pred == target {
                // True positive for this class
                *self.true_positives.entry(*target).or_insert(0) += 1;
            } else {
                // False negative for target class
                *self.false_negatives.entry(*target).or_insert(0) += 1;
            }
        }
    }

    fn compute(&self) -> f32 {
        if self.macro_average {
            let mut total_recall = 0.0;
            let mut valid_classes = 0;

            for class in 0..self.num_classes {
                let tp = self.true_positives.get(&class).unwrap_or(&0);
                let fn_count = self.false_negatives.get(&class).unwrap_or(&0);

                if tp + fn_count > 0 {
                    total_recall += *tp as f32 / (tp + fn_count) as f32;
                    valid_classes += 1;
                }
            }

            if valid_classes > 0 {
                total_recall / valid_classes as f32
            } else {
                0.0
            }
        } else {
            // Micro average (binary or overall)
            let total_tp: usize = self.true_positives.values().sum();
            let total_fn: usize = self.false_negatives.values().sum();

            if total_tp + total_fn == 0 {
                0.0
            } else {
                total_tp as f32 / (total_tp + total_fn) as f32
            }
        }
    }

    fn reset(&mut self) {
        self.true_positives.clear();
        self.false_negatives.clear();
    }

    fn name(&self) -> &str {
        if self.macro_average {
            "recall_macro"
        } else {
            "recall"
        }
    }
}

/// F1-score metric for classification tasks
#[derive(Debug, Clone)]
pub struct F1Score {
    /// Precision metric
    precision: Precision,
    /// Recall metric
    recall: Recall,
}

impl F1Score {
    /// Create a new F1-score metric
    pub fn new(num_classes: usize, macro_average: bool) -> Self {
        Self {
            precision: Precision::new(num_classes, macro_average),
            recall: Recall::new(num_classes, macro_average),
        }
    }

    /// Create a binary F1-score metric
    pub fn binary() -> Self {
        Self::new(2, false)
    }

    /// Create a macro-averaged F1-score metric
    pub fn macro_avg(num_classes: usize) -> Self {
        Self::new(num_classes, true)
    }
}

impl Metric for F1Score {
    type Prediction = usize;
    type Target = usize;

    fn update(&mut self, predictions: &[Self::Prediction], targets: &[Self::Target]) {
        self.precision.update(predictions, targets);
        self.recall.update(predictions, targets);
    }

    fn compute(&self) -> f32 {
        let p = self.precision.compute();
        let r = self.recall.compute();

        if p + r == 0.0 {
            0.0
        } else {
            2.0 * p * r / (p + r)
        }
    }

    fn reset(&mut self) {
        self.precision.reset();
        self.recall.reset();
    }

    fn name(&self) -> &str {
        if self.precision.macro_average {
            "f1_macro"
        } else {
            "f1"
        }
    }
}

/// Mean Squared Error metric for regression tasks
#[derive(Debug, Clone)]
pub struct MeanSquaredError {
    /// Sum of squared errors
    sum_squared_error: f32,
    /// Number of samples
    count: usize,
}

impl MeanSquaredError {
    /// Create a new MSE metric
    pub fn new() -> Self {
        Self {
            sum_squared_error: 0.0,
            count: 0,
        }
    }
}

impl Default for MeanSquaredError {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for MeanSquaredError {
    type Prediction = f32;
    type Target = f32;

    fn update(&mut self, predictions: &[Self::Prediction], targets: &[Self::Target]) {
        assert_eq!(predictions.len(), targets.len());

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            let error = pred - target;
            self.sum_squared_error += error * error;
            self.count += 1;
        }
    }

    fn compute(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_squared_error / self.count as f32
        }
    }

    fn reset(&mut self) {
        self.sum_squared_error = 0.0;
        self.count = 0;
    }

    fn name(&self) -> &str {
        "mse"
    }
}

/// Mean Absolute Error metric for regression tasks
#[derive(Debug, Clone)]
pub struct MeanAbsoluteError {
    /// Sum of absolute errors
    sum_absolute_error: f32,
    /// Number of samples
    count: usize,
}

impl MeanAbsoluteError {
    /// Create a new MAE metric
    pub fn new() -> Self {
        Self {
            sum_absolute_error: 0.0,
            count: 0,
        }
    }
}

impl Default for MeanAbsoluteError {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for MeanAbsoluteError {
    type Prediction = f32;
    type Target = f32;

    fn update(&mut self, predictions: &[Self::Prediction], targets: &[Self::Target]) {
        assert_eq!(predictions.len(), targets.len());

        for (pred, target) in predictions.iter().zip(targets.iter()) {
            self.sum_absolute_error += (pred - target).abs();
            self.count += 1;
        }
    }

    fn compute(&self) -> f32 {
        if self.count == 0 {
            0.0
        } else {
            self.sum_absolute_error / self.count as f32
        }
    }

    fn reset(&mut self) {
        self.sum_absolute_error = 0.0;
        self.count = 0;
    }

    fn name(&self) -> &str {
        "mae"
    }
}

/// Collection of metrics for tracking multiple metrics together
pub struct MetricsTracker {
    /// Map of metric name to metric implementation
    metrics: HashMap<String, Box<dyn MetricBox>>,
}

impl MetricsTracker {
    /// Create a new metrics tracker
    pub fn new() -> Self {
        Self {
            metrics: HashMap::new(),
        }
    }

    /// Add a metric to track
    pub fn add_metric<M>(&mut self, metric: M)
    where
        M: Metric + 'static,
        M::Prediction: 'static,
        M::Target: 'static,
    {
        let name = metric.name().to_string();
        self.metrics
            .insert(name, Box::new(MetricWrapper::new(metric)));
    }

    /// Update all metrics that can handle the given prediction/target types
    pub fn update<P, T>(&mut self, predictions: &[P], targets: &[T])
    where
        P: 'static + Clone,
        T: 'static + Clone,
    {
        for metric in self.metrics.values_mut() {
            metric.try_update_any(&predictions, &targets);
        }
    }

    /// Compute all metrics
    pub fn compute(&self) -> HashMap<String, f32> {
        self.metrics
            .iter()
            .map(|(name, metric)| (name.clone(), metric.compute()))
            .collect()
    }

    /// Reset all metrics
    pub fn reset(&mut self) {
        for metric in self.metrics.values_mut() {
            metric.reset();
        }
    }

    /// Get metric names
    pub fn metric_names(&self) -> Vec<String> {
        self.metrics.keys().cloned().collect()
    }

    /// Check if a metric exists
    pub fn has_metric(&self, name: &str) -> bool {
        self.metrics.contains_key(name)
    }
}

impl Default for MetricsTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait object wrapper for metrics
trait MetricBox {
    fn compute(&self) -> f32;
    fn reset(&mut self);
    fn name(&self) -> &str;
    fn try_update_any(&mut self, predictions: &dyn std::any::Any, targets: &dyn std::any::Any);
}

/// Wrapper for metric trait objects
struct MetricWrapper<M> {
    metric: M,
}

impl<M> MetricWrapper<M> {
    fn new(metric: M) -> Self {
        Self { metric }
    }
}

impl<M> MetricBox for MetricWrapper<M>
where
    M: Metric,
    M::Prediction: 'static,
    M::Target: 'static,
{
    fn compute(&self) -> f32 {
        self.metric.compute()
    }

    fn reset(&mut self) {
        self.metric.reset();
    }

    fn name(&self) -> &str {
        self.metric.name()
    }

    fn try_update_any(&mut self, predictions: &dyn std::any::Any, targets: &dyn std::any::Any) {
        if let (Some(preds), Some(targs)) = (
            predictions.downcast_ref::<Vec<M::Prediction>>(),
            targets.downcast_ref::<Vec<M::Target>>(),
        ) {
            self.metric.update(preds, targs);
        }
    }
}

/// Training metrics aggregator
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    /// Training loss history
    pub train_losses: Vec<f32>,
    /// Validation loss history
    pub val_losses: Vec<f32>,
    /// Training accuracy history
    pub train_accuracies: Vec<f32>,
    /// Validation accuracy history
    pub val_accuracies: Vec<f32>,
    /// Custom metrics history
    pub custom_metrics: HashMap<String, Vec<f32>>,
}

impl TrainingMetrics {
    /// Create a new training metrics aggregator
    pub fn new() -> Self {
        Self {
            train_losses: Vec::new(),
            val_losses: Vec::new(),
            train_accuracies: Vec::new(),
            val_accuracies: Vec::new(),
            custom_metrics: HashMap::new(),
        }
    }

    /// Add training loss
    pub fn add_train_loss(&mut self, loss: f32) {
        self.train_losses.push(loss);
    }

    /// Add validation loss
    pub fn add_val_loss(&mut self, loss: f32) {
        self.val_losses.push(loss);
    }

    /// Add training accuracy
    pub fn add_train_accuracy(&mut self, accuracy: f32) {
        self.train_accuracies.push(accuracy);
    }

    /// Add validation accuracy
    pub fn add_val_accuracy(&mut self, accuracy: f32) {
        self.val_accuracies.push(accuracy);
    }

    /// Add custom metric
    pub fn add_custom_metric(&mut self, name: &str, value: f32) {
        self.custom_metrics
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(value);
    }

    /// Get the latest training loss
    pub fn latest_train_loss(&self) -> Option<f32> {
        self.train_losses.last().copied()
    }

    /// Get the latest validation loss
    pub fn latest_val_loss(&self) -> Option<f32> {
        self.val_losses.last().copied()
    }

    /// Get the best validation loss
    pub fn best_val_loss(&self) -> Option<f32> {
        self.val_losses
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Get the best validation accuracy
    pub fn best_val_accuracy(&self) -> Option<f32> {
        self.val_accuracies
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
    }

    /// Clear all metrics
    pub fn clear(&mut self) {
        self.train_losses.clear();
        self.val_losses.clear();
        self.train_accuracies.clear();
        self.val_accuracies.clear();
        self.custom_metrics.clear();
    }
}

impl Default for TrainingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Alias for commonly used metrics collection
pub type Metrics = MetricsTracker;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_metric() {
        let mut accuracy = Accuracy::new();
        assert_eq!(accuracy.compute(), 0.0);

        let predictions = vec![0, 1, 1, 0, 1];
        let targets = vec![0, 1, 0, 0, 1];
        accuracy.update(&predictions, &targets);

        assert_eq!(accuracy.correct(), 4);
        assert_eq!(accuracy.total(), 5);
        assert_eq!(accuracy.compute(), 0.8);

        accuracy.reset();
        assert_eq!(accuracy.compute(), 0.0);
        assert_eq!(accuracy.correct(), 0);
        assert_eq!(accuracy.total(), 0);
    }

    #[test]
    fn test_precision_metric() {
        let mut precision = Precision::binary();

        let predictions = vec![1, 1, 0, 1, 0];
        let targets = vec![1, 0, 0, 1, 0];
        precision.update(&predictions, &targets);

        // TP = 2 (indices 0, 3), FP = 1 (index 1)
        // Precision = 2 / (2 + 1) = 0.666...
        assert!((precision.compute() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_recall_metric() {
        let mut recall = Recall::binary();

        let predictions = vec![1, 1, 0, 1, 0];
        let targets = vec![1, 0, 0, 1, 1];
        recall.update(&predictions, &targets);

        // TP = 2 (indices 0, 3), FN = 1 (index 4)
        // Recall = 2 / (2 + 1) = 0.666...
        assert!((recall.compute() - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_f1_score_metric() {
        let mut f1 = F1Score::binary();

        let predictions = vec![1, 1, 0, 1];
        let targets = vec![1, 0, 0, 1];
        f1.update(&predictions, &targets);

        let precision = 2.0 / 3.0; // TP=2, FP=1
        let recall = 1.0; // TP=2, FN=0
        let expected_f1 = 2.0 * precision * recall / (precision + recall);
        assert!((f1.compute() - expected_f1).abs() < 1e-6);
    }

    #[test]
    fn test_mse_metric() {
        let mut mse = MeanSquaredError::new();
        assert_eq!(mse.compute(), 0.0);

        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 2.5];
        mse.update(&predictions, &targets);

        // Errors: -0.5, -0.5, 0.5
        // Squared: 0.25, 0.25, 0.25
        // Mean: 0.25
        assert_eq!(mse.compute(), 0.25);
    }

    #[test]
    fn test_mae_metric() {
        let mut mae = MeanAbsoluteError::new();
        assert_eq!(mae.compute(), 0.0);

        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 2.5];
        mae.update(&predictions, &targets);

        // Absolute errors: 0.5, 0.5, 0.5
        // Mean: 0.5
        assert_eq!(mae.compute(), 0.5);
    }

    #[test]
    fn test_metrics_tracker() {
        let mut tracker = MetricsTracker::new();
        tracker.add_metric(Accuracy::new());
        tracker.add_metric(MeanSquaredError::new());

        assert!(tracker.has_metric("accuracy"));
        assert!(tracker.has_metric("mse"));
        assert!(!tracker.has_metric("precision"));

        let names = tracker.metric_names();
        assert!(names.contains(&"accuracy".to_string()));
        assert!(names.contains(&"mse".to_string()));

        // Test updating with classification data
        let class_preds = vec![0usize, 1, 1];
        let class_targets = vec![0usize, 1, 0];
        tracker.update(&class_preds, &class_targets);

        // Test updating with regression data
        let reg_preds = vec![1.0f32, 2.0, 3.0];
        let reg_targets = vec![1.5f32, 2.0, 2.5];
        tracker.update(&reg_preds, &reg_targets);

        let results = tracker.compute();
        assert!(results.contains_key("accuracy"));
        assert!(results.contains_key("mse"));

        tracker.reset();
        let results_after_reset = tracker.compute();
        assert_eq!(results_after_reset["accuracy"], 0.0);
        assert_eq!(results_after_reset["mse"], 0.0);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new();

        metrics.add_train_loss(0.5);
        metrics.add_train_loss(0.3);
        metrics.add_val_loss(0.4);
        metrics.add_val_loss(0.2);
        metrics.add_train_accuracy(0.8);
        metrics.add_val_accuracy(0.85);

        assert_eq!(metrics.latest_train_loss(), Some(0.3));
        assert_eq!(metrics.latest_val_loss(), Some(0.2));
        assert_eq!(metrics.best_val_loss(), Some(0.2));
        assert_eq!(metrics.best_val_accuracy(), Some(0.85));

        metrics.add_custom_metric("f1", 0.9);
        metrics.add_custom_metric("f1", 0.95);
        assert_eq!(metrics.custom_metrics["f1"], vec![0.9, 0.95]);

        metrics.clear();
        assert!(metrics.train_losses.is_empty());
        assert!(metrics.custom_metrics.is_empty());
    }
}
