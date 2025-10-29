//! Tensor operations

use crate::backend::Backend;
use crate::data::{DataType, Element, Float};
use crate::shape::Shape;
use crate::tensor::Tensor;

/// Basic arithmetic operations
impl<B: Backend + Default, const D: usize> Tensor<B, D, Float>
where
    <Float as DataType>::Primitive: Element,
{
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch for addition");

        let self_data = self.to_data();
        let other_data = other.to_data();

        let result_data: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch for subtraction"
        );

        let self_data = self.to_data();
        let other_data = other.to_data();

        let result_data: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape(),
            other.shape(),
            "Shape mismatch for multiplication"
        );

        let self_data = self.to_data();
        let other_data = other.to_data();

        let result_data: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise division
    pub fn div(&self, other: &Self) -> Self {
        assert_eq!(self.shape(), other.shape(), "Shape mismatch for division");

        let self_data = self.to_data();
        let other_data = other.to_data();

        let result_data: Vec<f32> = self_data
            .iter()
            .zip(other_data.iter())
            .map(|(a, b)| a / b)
            .collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Scalar addition
    pub fn add_scalar(&self, scalar: f32) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|a| a + scalar).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Scalar subtraction
    pub fn sub_scalar(&self, scalar: f32) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|a| a - scalar).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Scalar multiplication
    pub fn mul_scalar(&self, scalar: f32) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|a| a * scalar).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Scalar division
    pub fn div_scalar(&self, scalar: f32) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|a| a / scalar).collect();

        Self::from_data(result_data, self.shape().clone())
    }
}

/// Matrix multiplication for 2D tensors
impl<B: Backend + Default> Tensor<B, 2, Float>
where
    <Float as DataType>::Primitive: Element,
{
    /// Matrix multiplication (for 2D tensors)
    pub fn matmul(&self, other: &Self) -> Self {
        assert_eq!(
            self.shape().dim(1),
            other.shape().dim(0),
            "Inner dimensions must match for matrix multiplication"
        );

        let m = self.shape().dim(0);
        let n = other.shape().dim(1);
        let k = self.shape().dim(1);

        let self_data = self.to_data();
        let other_data = other.to_data();
        let mut result_data = vec![0.0; m * n];

        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    result_data[i * n + j] += self_data[i * k + l] * other_data[l * n + j];
                }
            }
        }

        let result_shape = Shape::new([m, n]);
        Self::from_data(result_data, result_shape)
    }
}

/// Advanced mathematical operations
impl<B: Backend + Default, const D: usize> Tensor<B, D, Float>
where
    <Float as DataType>::Primitive: Element,
{
    /// Element-wise sine
    pub fn sin(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|x| x.sin()).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise cosine
    pub fn cos(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|x| x.cos()).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|x| x.exp()).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise natural logarithm
    pub fn ln(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|x| x.ln()).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|x| x.sqrt()).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise power
    pub fn pow(&self, exponent: f32) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|x| x.powf(exponent)).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|x| x.abs()).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Sum all elements
    pub fn sum(&self) -> f32 {
        self.to_data().iter().sum()
    }

    /// Mean of all elements
    pub fn mean(&self) -> f32 {
        let data = self.to_data();
        data.iter().sum::<f32>() / data.len() as f32
    }

    /// Maximum element
    pub fn max(&self) -> f32 {
        self.to_data()
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b))
    }

    /// Minimum element
    pub fn min(&self) -> f32 {
        self.to_data().iter().fold(f32::INFINITY, |a, &b| a.min(b))
    }
}

/// Activation functions
impl<B: Backend + Default, const D: usize> Tensor<B, D, Float>
where
    <Float as DataType>::Primitive: Element,
{
    /// ReLU activation function (Rectified Linear Unit)
    /// f(x) = max(0, x)
    pub fn relu(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|&x| x.max(0.0)).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Leaky ReLU activation function
    /// f(x) = max(alpha * x, x) where alpha is typically 0.01
    pub fn leaky_relu(&self, alpha: f32) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data
            .iter()
            .map(|&x| if x >= 0.0 { x } else { alpha * x })
            .collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Sigmoid activation function
    /// f(x) = 1 / (1 + exp(-x))
    pub fn sigmoid(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Tanh activation function
    /// f(x) = tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    pub fn tanh(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|&x| x.tanh()).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Softmax activation function (applied along the last dimension)
    /// For 2D tensors, applies softmax to each row
    /// For 1D tensors, applies softmax to the entire vector
    pub fn softmax(&self) -> Self {
        match D {
            1 => self.softmax_1d(),
            2 => self.softmax_2d(),
            _ => panic!("Softmax only implemented for 1D and 2D tensors"),
        }
    }

    /// Helper function for 1D softmax
    fn softmax_1d(&self) -> Self {
        let self_data = self.to_data();

        // Find max for numerical stability
        let max_val = self_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) for numerical stability
        let exp_vals: Vec<f32> = self_data.iter().map(|&x| (x - max_val).exp()).collect();

        // Compute sum of exponentials
        let sum_exp: f32 = exp_vals.iter().sum();

        // Normalize
        let result_data: Vec<f32> = exp_vals.iter().map(|&x| x / sum_exp).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Helper function for 2D softmax (applies softmax to each row)
    fn softmax_2d(&self) -> Self {
        let self_data = self.to_data();
        let rows = self.shape().dim(0);
        let cols = self.shape().dim(1);
        let mut result_data = vec![0.0; rows * cols];

        for i in 0..rows {
            let row_start = i * cols;
            let row_end = row_start + cols;
            let row_data = &self_data[row_start..row_end];

            // Find max for numerical stability
            let max_val = row_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            // Compute exp(x - max) for numerical stability
            let exp_vals: Vec<f32> = row_data.iter().map(|&x| (x - max_val).exp()).collect();

            // Compute sum of exponentials
            let sum_exp: f32 = exp_vals.iter().sum();

            // Normalize and store
            for (j, &exp_val) in exp_vals.iter().enumerate() {
                result_data[row_start + j] = exp_val / sum_exp;
            }
        }

        Self::from_data(result_data, self.shape().clone())
    }

    /// GELU activation function (Gaussian Error Linear Unit)
    /// f(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    pub fn gelu(&self) -> Self {
        let self_data = self.to_data();
        let sqrt_2_over_pi = (2.0 / std::f32::consts::PI).sqrt();

        let result_data: Vec<f32> = self_data
            .iter()
            .map(|&x| {
                let inner = sqrt_2_over_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            })
            .collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// Swish/SiLU activation function
    /// f(x) = x * sigmoid(x) = x / (1 + exp(-x))
    pub fn swish(&self) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

        Self::from_data(result_data, self.shape().clone())
    }

    /// ELU activation function (Exponential Linear Unit)
    /// f(x) = x if x >= 0, alpha * (exp(x) - 1) if x < 0
    pub fn elu(&self, alpha: f32) -> Self {
        let self_data = self.to_data();
        let result_data: Vec<f32> = self_data
            .iter()
            .map(|&x| if x >= 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect();

        Self::from_data(result_data, self.shape().clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;

    #[test]
    fn test_element_wise_operations() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());
        let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], shape);

        let sum = a.add(&b);
        assert_eq!(sum.to_data(), vec![6.0, 8.0, 10.0, 12.0]);

        let diff = a.sub(&b);
        assert_eq!(diff.to_data(), vec![-4.0, -4.0, -4.0, -4.0]);

        let prod = a.mul(&b);
        assert_eq!(prod.to_data(), vec![5.0, 12.0, 21.0, 32.0]);

        let div = a.div(&b);
        let expected = vec![1.0 / 5.0, 2.0 / 6.0, 3.0 / 7.0, 4.0 / 8.0];
        let result = div.to_data();
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_scalar_operations() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);

        let add_result = a.add_scalar(10.0);
        assert_eq!(add_result.to_data(), vec![11.0, 12.0, 13.0, 14.0]);

        let sub_result = a.sub_scalar(1.0);
        assert_eq!(sub_result.to_data(), vec![0.0, 1.0, 2.0, 3.0]);

        let mul_result = a.mul_scalar(2.0);
        assert_eq!(mul_result.to_data(), vec![2.0, 4.0, 6.0, 8.0]);

        let div_result = a.div_scalar(2.0);
        assert_eq!(div_result.to_data(), vec![0.5, 1.0, 1.5, 2.0]);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
        let b: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2]));

        let result = a.matmul(&b);
        // [1 2] * [5 6] = [19 22]
        // [3 4]   [7 8]   [43 50]
        assert_eq!(result.to_data(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matrix_multiplication_rectangular() {
        let a: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3]));
        let b: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], Shape::new([3, 2]));

        let result = a.matmul(&b);
        // [1 2 3] * [7  8 ] = [58  64]
        // [4 5 6]   [9  10]   [139 154]
        //           [11 12]
        assert_eq!(result.to_data(), vec![58.0, 64.0, 139.0, 154.0]);
        assert_eq!(result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_mathematical_operations() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.0, 1.0, 2.0, 3.0], shape);

        let sin_result = a.sin();
        let cos_result = a.cos();
        let exp_result = a.exp();

        // Check sin(0) = 0, sin(1) ≈ 0.841
        assert!((sin_result.to_data()[0] - 0.0).abs() < 1e-6);
        assert!((sin_result.to_data()[1] - 1.0_f32.sin()).abs() < 1e-6);

        // Check cos(0) = 1
        assert!((cos_result.to_data()[0] - 1.0).abs() < 1e-6);

        // Check exp(0) = 1
        assert!((exp_result.to_data()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_reduction_operations() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);

        assert_eq!(a.sum(), 10.0);
        assert_eq!(a.mean(), 2.5);
        assert_eq!(a.max(), 4.0);
        assert_eq!(a.min(), 1.0);
    }

    #[test]
    fn test_operation_chaining() {
        let a: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
        let b: Tensor<CpuBackend, 2> =
            Tensor::from_data(vec![1.0, 1.0, 1.0, 1.0], Shape::new([2, 2]));

        // Chain operations: (a + b) * a
        let result = a.add(&b).mul(&a);

        // a + b = [2, 3, 4, 5]
        // (a + b) * a = [2*1, 3*2, 4*3, 5*4] = [2, 6, 12, 20]
        assert_eq!(result.to_data(), vec![2.0, 6.0, 12.0, 20.0]);
    }

    #[test]
    #[should_panic(expected = "Shape mismatch")]
    fn test_shape_mismatch_operations() {
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0], Shape::new([2, 1]));
        let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0], Shape::new([3, 1]));

        // This should panic due to shape mismatch
        let _ = a.add(&b);
    }

    #[test]
    #[should_panic(expected = "Inner dimensions must match")]
    fn test_invalid_matrix_multiplication() {
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0], Shape::new([1, 2]));
        let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0], Shape::new([3, 1]));

        // This should panic: [1, 2] @ [3, 1] - inner dimensions don't match
        let _ = a.matmul(&b);
    }

    #[test]
    fn test_relu_activation() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], shape);

        let result = a.relu();
        assert_eq!(result.to_data(), vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_leaky_relu_activation() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], shape);

        let result = a.leaky_relu(0.1);
        assert_eq!(result.to_data(), vec![-0.1, 2.0, -0.3, 4.0]);
    }

    #[test]
    fn test_sigmoid_activation() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.0, 1.0, -1.0, 2.0], shape);

        let result = a.sigmoid();
        let data = result.to_data();

        // sigmoid(0) = 0.5
        assert!((data[0] - 0.5).abs() < 1e-6);
        // sigmoid(1) ≈ 0.7311
        assert!((data[1] - 0.7310586).abs() < 1e-6);
        // sigmoid(-1) ≈ 0.2689
        assert!((data[2] - 0.26894143).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_activation() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.0, 1.0, -1.0, 2.0], shape);

        let result = a.tanh();
        let data = result.to_data();

        // tanh(0) = 0
        assert!((data[0] - 0.0).abs() < 1e-6);
        // tanh(1) ≈ 0.7616
        assert!((data[1] - 1.0_f32.tanh()).abs() < 1e-6);
        // tanh(-1) ≈ -0.7616
        assert!((data[2] - (-1.0_f32).tanh()).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_1d() {
        let shape = Shape::new([4]);
        let a: Tensor<CpuBackend, 1> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);

        let result = a.softmax();
        let data = result.to_data();

        // Check that probabilities sum to 1
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all values are positive
        for &val in &data {
            assert!(val > 0.0);
        }

        // Check that the largest input corresponds to the largest output
        let max_index = data
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_index, 3); // index of max input (4.0)
    }

    #[test]
    fn test_softmax_2d() {
        let shape = Shape::new([2, 3]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape);

        let result = a.softmax();
        let data = result.to_data();

        // Check that each row sums to 1
        let row1_sum: f32 = data[0..3].iter().sum();
        let row2_sum: f32 = data[3..6].iter().sum();
        assert!((row1_sum - 1.0).abs() < 1e-6);
        assert!((row2_sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_gelu_activation() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.0, 1.0, -1.0, 2.0], shape);

        let result = a.gelu();
        let data = result.to_data();

        // GELU(0) = 0
        assert!((data[0] - 0.0).abs() < 1e-6);
        // GELU should be smooth and non-zero for positive inputs
        assert!(data[1] > 0.0);
        assert!(data[3] > 0.0);
    }

    #[test]
    fn test_swish_activation() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.0, 1.0, -1.0, 2.0], shape);

        let result = a.swish();
        let data = result.to_data();

        // Swish(0) = 0
        assert!((data[0] - 0.0).abs() < 1e-6);
        // Swish should be positive for positive inputs
        assert!(data[1] > 0.0);
        assert!(data[3] > 0.0);
    }

    #[test]
    fn test_elu_activation() {
        let shape = Shape::new([2, 2]);
        let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], shape);

        let result = a.elu(1.0);
        let data = result.to_data();

        // ELU preserves positive values
        assert_eq!(data[1], 2.0);
        assert_eq!(data[3], 4.0);

        // ELU transforms negative values
        assert!(data[0] < 0.0 && data[0] > -1.0); // Should be between -1 and 0
        assert!(data[2] < 0.0 && data[2] > -1.0);
    }
}
