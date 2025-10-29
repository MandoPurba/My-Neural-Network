//! Mini-Burn: A minimal deep learning framework inspired by Burn
//!
//! This framework provides a tensor abstraction with support for multiple backends,
//! data types, and multi-dimensional operations.

pub mod autodiff;
pub mod backend;
pub mod broadcast;
pub mod data;
pub mod error;
pub mod nn;
pub mod ops;
pub mod optim;
pub mod shape;
pub mod tensor;
// pub mod train;  // Temporarily commented out due to compilation errors

// Re-export main types for convenience
pub use autodiff::{AutodiffTensor, GradTensor, RequiresGrad};
pub use backend::{Backend, CpuBackend};
pub use broadcast::{BroadcastError, Broadcastable, Broadcasting};
pub use data::{Bool, Float, Int};
pub use error::{MiniBurnError, Result};
pub use optim::{Adam, Optimizer, Parameter, Sgd, TensorParameter};
pub use shape::Shape;
pub use tensor::Tensor;
// pub use train::{TrainingConfig, TrainingLoop, TrainingMetrics}; // Temporarily commented out

/// The default float type used throughout the framework
pub type DefaultFloat = f32;

/// The default int type used throughout the framework
pub type DefaultInt = i32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_tensor_creation() {
        // Test basic tensor creation with CPU backend
        let shape = Shape::new([2, 3]);

        // Create a float tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(data, shape);

        assert_eq!(tensor.shape().dims(), &[2, 3]);
    }

    #[test]
    fn tensor_with_different_types() {
        let shape = Shape::new([2, 2]);

        // Float tensor (default)
        let float_data = vec![1.0, 2.0, 3.0, 4.0];
        let float_tensor: Tensor<CpuBackend, 2> = Tensor::from_data(float_data, shape.clone());

        // Explicit float tensor
        let explicit_float_tensor: Tensor<CpuBackend, 2, Float> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());

        // Int tensor
        let int_data = vec![1, 2, 3, 4];
        let int_tensor: Tensor<CpuBackend, 2, Int> = Tensor::from_data(int_data, shape.clone());

        // Bool tensor
        let bool_data = vec![true, false, true, false];
        let bool_tensor: Tensor<CpuBackend, 2, Bool> = Tensor::from_data(bool_data, shape);

        assert_eq!(float_tensor.shape().dims(), &[2, 2]);
        assert_eq!(explicit_float_tensor.shape().dims(), &[2, 2]);
        assert_eq!(int_tensor.shape().dims(), &[2, 2]);
        assert_eq!(bool_tensor.shape().dims(), &[2, 2]);
    }
}
