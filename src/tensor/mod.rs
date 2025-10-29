//! Core tensor implementation

use crate::backend::Backend;
use crate::data::{DataType, Float};
use crate::shape::Shape;
use std::marker::PhantomData;

/// Main tensor struct with generic backend, dimension, and data type
pub struct Tensor<B: Backend, const D: usize, K: DataType = Float> {
    /// Storage backend
    storage: B::Storage<K>,
    /// Shape of the tensor
    shape: Shape<D>,
    /// Backend instance
    backend: B,
    /// Phantom data for type tracking
    _phantom: PhantomData<K>,
}

impl<B: Backend + Default, const D: usize, K: DataType> Tensor<B, D, K> {
    /// Create a new tensor from data and shape
    pub fn from_data(data: Vec<K::Primitive>, shape: Shape<D>) -> Self {
        assert_eq!(
            data.len(),
            shape.numel(),
            "Data length {} doesn't match shape size {}",
            data.len(),
            shape.numel()
        );

        let backend = B::default();
        let device = B::default_device();
        let storage = B::from_data(data, &device);

        Self {
            storage,
            shape,
            backend,
            _phantom: PhantomData,
        }
    }

    /// Get the shape of the tensor
    pub fn shape(&self) -> &Shape<D> {
        &self.shape
    }

    /// Create a tensor of zeros with the same shape as this tensor
    pub fn zeros_like(&self) -> Self
    where
        K::Primitive: Clone + Default,
    {
        let zero_value = K::Primitive::default();
        let size = self.shape.numel();
        let data = vec![zero_value; size];
        Tensor::from_data(data, self.shape.clone())
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }

    /// Convert tensor data back to Vec
    pub fn to_data(&self) -> Vec<K::Primitive> {
        B::to_data(&self.storage)
    }

    /// Get a reference to the backend
    pub fn backend(&self) -> &B {
        &self.backend
    }
}

impl<B: Backend, const D: usize, K: DataType> Clone for Tensor<B, D, K> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            backend: self.backend.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<B: Backend, const D: usize, K: DataType> std::fmt::Debug for Tensor<B, D, K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor<{}, {}> {}", K::name(), D, self.shape)
    }
}

// Convenience constructors for different types
impl<B: Backend + Default, const D: usize> Tensor<B, D, Float> {
    /// Create a tensor filled with zeros
    pub fn zeros(shape: Shape<D>) -> Self {
        let data = vec![0.0; shape.numel()];
        Self::from_data(data, shape)
    }

    /// Create a tensor filled with ones
    pub fn ones(shape: Shape<D>) -> Self {
        let data = vec![1.0; shape.numel()];
        Self::from_data(data, shape)
    }

    /// Create a tensor filled with a specific value
    pub fn full(shape: Shape<D>, value: f32) -> Self {
        let data = vec![value; shape.numel()];
        Self::from_data(data, shape)
    }
}

// Special constructor for 1D tensors
impl<B: Backend + Default> Tensor<B, 1, Float> {
    /// Create a range tensor (1D only)
    pub fn range(start: f32, end: f32, step: f32) -> Self {
        let mut data = Vec::new();
        let mut current = start;
        while current < end {
            data.push(current);
            current += step;
        }
        let shape = Shape::new([data.len()]);
        Self::from_data(data, shape)
    }
}

// Type aliases for convenience
pub type FloatTensor<B, const D: usize> = Tensor<B, D, Float>;
pub type IntTensor<B, const D: usize> = Tensor<B, D, crate::data::Int>;
pub type BoolTensor<B, const D: usize> = Tensor<B, D, crate::data::Bool>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::CpuBackend;
    use crate::data::{Bool, Int};

    #[test]
    fn test_tensor_creation() {
        let shape = Shape::new([2, 3]);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(data, shape);

        assert_eq!(tensor.numel(), 6);
        assert_eq!(tensor.ndim(), 2);
        assert_eq!(tensor.shape().dims(), &[2, 3]);
    }

    #[test]
    fn test_tensor_zeros_ones() {
        let shape = Shape::new([3, 3]);

        let zeros: Tensor<CpuBackend, 2> = Tensor::zeros(shape.clone());
        let ones: Tensor<CpuBackend, 2> = Tensor::ones(shape);

        assert_eq!(zeros.to_data(), vec![0.0; 9]);
        assert_eq!(ones.to_data(), vec![1.0; 9]);
    }

    #[test]
    fn test_tensor_full() {
        let shape = Shape::new([2, 2]);
        let filled: Tensor<CpuBackend, 2> = Tensor::full(shape, 42.0);

        assert_eq!(filled.to_data(), vec![42.0; 4]);
    }

    #[test]
    fn test_tensor_range() {
        let range: Tensor<CpuBackend, 1> = Tensor::range(0.0, 5.0, 1.0);
        assert_eq!(range.to_data(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

        let range_step: Tensor<CpuBackend, 1> = Tensor::range(0.0, 3.0, 0.5);
        assert_eq!(range_step.to_data(), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
    }

    #[test]
    fn test_different_data_types() {
        let shape = Shape::new([2, 2]);

        let float_tensor: Tensor<CpuBackend, 2, Float> =
            Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());

        let int_tensor: Tensor<CpuBackend, 2, Int> =
            Tensor::from_data(vec![1, 2, 3, 4], shape.clone());

        let bool_tensor: Tensor<CpuBackend, 2, Bool> =
            Tensor::from_data(vec![true, false, true, false], shape);

        assert_eq!(float_tensor.numel(), 4);
        assert_eq!(int_tensor.numel(), 4);
        assert_eq!(bool_tensor.numel(), 4);
    }

    #[test]
    fn test_tensor_clone() {
        let shape = Shape::new([2, 2]);
        let original: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);
        let cloned = original.clone();

        assert_eq!(original.to_data(), cloned.to_data());
        assert_eq!(original.shape(), cloned.shape());
    }

    #[test]
    fn test_tensor_debug() {
        let shape = Shape::new([2, 2]);
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape);

        let debug_str = format!("{:?}", tensor);
        assert!(debug_str.contains("Tensor<Float, 2>"));
        assert!(debug_str.contains("[2, 2]"));
    }

    #[test]
    #[should_panic(expected = "Data length")]
    fn test_invalid_data_length() {
        let shape = Shape::new([2, 2]);
        let _tensor: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0], shape);
        // Need 4 elements
    }
}
