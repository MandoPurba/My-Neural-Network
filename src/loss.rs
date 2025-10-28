/// Loss Functions - Mengukur seberapa salah prediksi kita
/// Ini adalah "guru" yang memberitahu NN arah mana yang harus diperbaiki
use crate::Matrix;

pub trait LossFunction {
    /// Menghitung loss (seberapa salah prediksi)
    fn calculate_loss(&self, predictions: &Matrix, targets: &Matrix) -> f32;

    /// Menghitung gradient loss untuk backpropagation
    fn calculate_gradient(&self, predictions: &Matrix, targets: &Matrix) -> Matrix;
}

/// Mean Squared Error - Bagus untuk regression problems
/// L = (1/n) * Σ(predicted - actual)²
pub struct MeanSquaredError;

impl LossFunction for MeanSquaredError {
    fn calculate_loss(&self, predictions: &Matrix, targets: &Matrix) -> f32 {
        assert_eq!(
            predictions.rows, targets.rows,
            "Predictions and targets must have same rows"
        );
        assert_eq!(
            predictions.cols, targets.cols,
            "Predictions and targets must have same cols"
        );

        let mut sum = 0.0;
        let total_elements = predictions.data.len() as f32;

        for i in 0..predictions.data.len() {
            let diff = predictions.data[i] - targets.data[i];
            sum += diff * diff;
        }

        sum / total_elements
    }

    /// Gradient of MSE: dL/dy = 2(predicted - actual) / n
    fn calculate_gradient(&self, predictions: &Matrix, targets: &Matrix) -> Matrix {
        assert_eq!(
            predictions.rows, targets.rows,
            "Predictions and targets must have same rows"
        );
        assert_eq!(
            predictions.cols, targets.cols,
            "Predictions and targets must have same cols"
        );

        let n = predictions.data.len() as f32;
        let diff = predictions.subtract(targets);
        diff.scale(2.0 / n)
    }
}

/// Binary Cross Entropy - Bagus untuk binary classification
/// L = -(1/n) * Σ[y*log(p) + (1-y)*log(1-p)]
pub struct BinaryCrossEntropy;

impl LossFunction for BinaryCrossEntropy {
    fn calculate_loss(&self, predictions: &Matrix, targets: &Matrix) -> f32 {
        assert_eq!(
            predictions.rows, targets.rows,
            "Predictions and targets must have same rows"
        );
        assert_eq!(
            predictions.cols, targets.cols,
            "Predictions and targets must have same cols"
        );

        let mut sum = 0.0;
        let total_elements = predictions.data.len() as f32;

        for i in 0..predictions.data.len() {
            let y = targets.data[i];
            let p = predictions.data[i].max(1e-15).min(1.0 - 1e-15); // Clamp untuk avoid log(0)

            sum += -(y * p.ln() + (1.0 - y) * (1.0 - p).ln());
        }

        sum / total_elements
    }

    /// Gradient of Binary Cross Entropy: dL/dy = (p - y) / (p * (1 - p))
    fn calculate_gradient(&self, predictions: &Matrix, targets: &Matrix) -> Matrix {
        assert_eq!(
            predictions.rows, targets.rows,
            "Predictions and targets must have same rows"
        );
        assert_eq!(
            predictions.cols, targets.cols,
            "Predictions and targets must have same cols"
        );

        let mut result = Matrix::new(predictions.rows, predictions.cols);
        let n = predictions.data.len() as f32;

        for i in 0..predictions.data.len() {
            let y = targets.data[i];
            let p = predictions.data[i].max(1e-15).min(1.0 - 1e-15); // Clamp untuk avoid division by 0

            result.data[i] = (p - y) / (p * (1.0 - p) * n);
        }

        result
    }
}
