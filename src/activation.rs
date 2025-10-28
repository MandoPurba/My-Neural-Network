/// Acticavation FUnctuons - Memberikan non-linearitas ke Neural Network
/// Tampa ini, NN hanya bisa belajar fungsi linear sederhana

/// Trait untuk semua activation functions
pub trait ActivationFunction {
    /// Forward pass: menerapkan activation function
    fn activate(&self, x: f32) -> f32;
    /// Backward pass: menghitung turunan activation function
    fn derivative(&self, x: f32) -> f32;
}

/// ReLU (Rectified Linear Unit) - Paling populer di deep learning
/// f(x) = max(0, x)
/// Sederhana tapi sangat efektif!
pub struct ReLU;

impl ActivationFunction for ReLU {
    fn activate(&self, x: f32) -> f32 {
        if x > 0. { x } else { 0. }
    }

    /// Derivative of ReLU
    /// f'(x) = 1 if x > 0 else 0
    fn derivative(&self, x: f32) -> f32 {
        if x > 0. { 1. } else { 0. }
    }
}

/// Sigmoid - Menghasilkan output antara 0 dan 1
/// f(x) = 1 / (1 + exp(-x))
/// Bagus untuk binary classification
pub struct Sigmoid;

impl ActivationFunction for Sigmoid {
    fn activate(&self, x: f32) -> f32 {
        1. / (1. + (-x).exp())
    }

    /// Derivative of Sigmoid
    /// f'(x) = f(x) * (1 - f(x))
    fn derivative(&self, x: f32) -> f32 {
        let s = self.activate(x);
        s * (1. - s)
    }
}

/// Tanh (Hyperbolic Tangent) - Output antara -1 dan 1
/// f(x) = tanh(x)
/// Centered arround zero, kadang lebih baik dari Sigmoid
pub struct Tanh;

impl ActivationFunction for Tanh {
    fn activate(&self, x: f32) -> f32 {
        x.tanh()
    }

    /// Derivative of Tanh
    /// f'(x) = 1 - tanh^2(x)
    fn derivative(&self, x: f32) -> f32 {
        let t = self.activate(x);
        1. - t * t
    }
}

/// Linear activation - tidak mengubah input
/// f(x) = x
/// Biasanya digunakan di output layer untuk regression
pub struct Linear;

impl ActivationFunction for Linear {
    fn activate(&self, x: f32) -> f32 {
        x
    }

    fn derivative(&self, _x: f32) -> f32 {
        1.
    }
}

/// Softmax - Mengubah vector menjadi probabilitas
/// f(x_i) = exp(x_i) / sum(exp(x_j))
/// Sering digunakan di output layer untuk multi-class classification
pub struct Softmax;

impl Softmax {
    /// Apply softmax to a slice of inputs
    pub fn activate_vector(&self, inputs: &[f32]) -> Vec<f32> {
        let max_input = inputs.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_values: Vec<f32> = inputs.iter().map(|&x| (x - max_input).exp()).collect();
        let sum_exp: f32 = exp_values.iter().sum();
        exp_values.iter().map(|&v| v / sum_exp).collect()
    }

    /// Derivative of softmax is more complex and usually handled with cross-entropy loss
    /// Here we provide a placeholder
    pub fn derivative_vector(&self, _inputs: &[f32]) -> Vec<f32> {
        // Placeholder implementation
        vec![]
    }
}

/// Helper functions untuk  menerapkan activation ke matrix
use crate::Matrix;

/// Menerapkan activation function ke seluruh matrix
/// Contoh penggunaan:
/// let input = Matrix::from_data(vec![-1.0, 0.0, 1.0, 2.0], 2, 2);
/// let relu = ReLU;
/// let output = apply_activation(&input, &relu);
/// // Output akan menjadi [0.0, 0.0, 1.0, 2.0]
pub fn apply_activation(matrix: &Matrix, activation: &dyn ActivationFunction) -> Matrix {
    matrix.map(|x| activation.activate(x))
}
