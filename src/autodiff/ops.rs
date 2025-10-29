//! Autodiff Operations Module
//!
//! This module implements concrete gradient functions for various operations
//! used in automatic differentiation.

use crate::autodiff::tape::TapeOperation;
use crate::autodiff::NodeId;
use crate::backend::Backend;
use crate::data::DataType;
use crate::shape::Shape;
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Trait for operations that support automatic differentiation
pub trait AutodiffOp<B: Backend> {
    type Input;
    type Output;

    /// Execute the forward pass
    fn forward(&self, input: Self::Input) -> Self::Output;

    /// Create a tape operation for this autodiff operation
    fn create_tape_operation(&self, output_id: NodeId, input_ids: Vec<NodeId>) -> TapeOperation<B>;
}

/// Addition operation between two tensors
#[derive(Debug, Clone)]
pub struct AddOp<B: Backend, const D: usize, K: DataType> {
    pub lhs_shape: Shape<D>,
    pub rhs_shape: Shape<D>,
    _phantom: std::marker::PhantomData<(B, K)>,
}

impl<B: Backend, const D: usize, K: DataType> AddOp<B, D, K> {
    pub fn new(lhs_shape: Shape<D>, rhs_shape: Shape<D>) -> Self {
        Self {
            lhs_shape,
            rhs_shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> AutodiffOp<B> for AddOp<B, D, K>
where
    K::Primitive: Clone + std::ops::Add<Output = K::Primitive> + Send + Sync + 'static,
{
    type Input = (Tensor<B, D, K>, Tensor<B, D, K>);
    type Output = Tensor<B, D, K>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        let (lhs, rhs) = input;
        // Delegate to existing tensor add operation
        if D == 1 {
            // For now, implement simple element-wise addition
            let lhs_data = lhs.to_data();
            let rhs_data = rhs.to_data();

            assert_eq!(
                lhs_data.len(),
                rhs_data.len(),
                "Tensor size mismatch for addition"
            );

            let result_data: Vec<K::Primitive> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a.clone() + b.clone())
                .collect();

            Tensor::from_data(result_data, lhs.shape().clone())
        } else {
            // For higher dimensions, we'd need proper broadcasting
            panic!("Broadcasting not yet implemented for dimensions > 1")
        }
    }

    fn create_tape_operation(&self, output_id: NodeId, input_ids: Vec<NodeId>) -> TapeOperation<B> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "lhs_shape".to_string(),
            format!("{:?}", self.lhs_shape.dims()),
        );
        metadata.insert(
            "rhs_shape".to_string(),
            format!("{:?}", self.rhs_shape.dims()),
        );

        TapeOperation {
            op_type: "Add".to_string(),
            output_id,
            input_ids,
            grad_fn: Some(std::sync::Arc::new(AddGradFn::<B, D, K>::new(
                self.lhs_shape.clone(),
                self.rhs_shape.clone(),
            ))),
            metadata,
        }
    }
}

/// Gradient function for addition
#[derive(Debug)]
pub struct AddGradFn<B: Backend, const D: usize, K: DataType> {
    _lhs_shape: Shape<D>,
    _rhs_shape: Shape<D>,
    _phantom: std::marker::PhantomData<(B, K)>,
}

impl<B: Backend, const D: usize, K: DataType> AddGradFn<B, D, K> {
    pub fn new(lhs_shape: Shape<D>, rhs_shape: Shape<D>) -> Self {
        Self {
            _lhs_shape: lhs_shape,
            _rhs_shape: rhs_shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> crate::autodiff::tape::TapeGradFn<B>
    for AddGradFn<B, D, K>
where
    K::Primitive: Clone + Send + Sync + 'static,
{
    fn compute_gradients(
        &self,
        grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>> {
        // For addition: grad_lhs = grad_output, grad_rhs = grad_output
        // If shapes differ, we need to sum over broadcasted dimensions
        if let Some(grad) = grad_output.downcast_ref::<Tensor<B, D, K>>() {
            // For now, assume same shapes (no broadcasting)
            vec![
                Box::new(grad.clone()) as Box<dyn std::any::Any + Send + Sync>,
                Box::new(grad.clone()) as Box<dyn std::any::Any + Send + Sync>,
            ]
        } else {
            vec![]
        }
    }

    fn description(&self) -> String {
        "AddGradFn".to_string()
    }
}

/// Multiplication operation between two tensors
#[derive(Debug, Clone)]
pub struct MulOp<B: Backend, const D: usize, K: DataType> {
    pub lhs: Tensor<B, D, K>,
    pub rhs: Tensor<B, D, K>,
}

impl<B: Backend, const D: usize, K: DataType> MulOp<B, D, K> {
    pub fn new(lhs: Tensor<B, D, K>, rhs: Tensor<B, D, K>) -> Self {
        Self { lhs, rhs }
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> AutodiffOp<B> for MulOp<B, D, K>
where
    K::Primitive: Clone + std::ops::Mul<Output = K::Primitive> + Send + Sync + 'static,
{
    type Input = ();
    type Output = Tensor<B, D, K>;

    fn forward(&self, _input: Self::Input) -> Self::Output {
        // Element-wise multiplication
        if D == 1 {
            let lhs_data = self.lhs.to_data();
            let rhs_data = self.rhs.to_data();

            assert_eq!(
                lhs_data.len(),
                rhs_data.len(),
                "Tensor size mismatch for multiplication"
            );

            let result_data: Vec<K::Primitive> = lhs_data
                .iter()
                .zip(rhs_data.iter())
                .map(|(a, b)| a.clone() * b.clone())
                .collect();

            Tensor::from_data(result_data, self.lhs.shape().clone())
        } else {
            panic!("Broadcasting not yet implemented for dimensions > 1")
        }
    }

    fn create_tape_operation(&self, output_id: NodeId, input_ids: Vec<NodeId>) -> TapeOperation<B> {
        TapeOperation {
            op_type: "Mul".to_string(),
            output_id,
            input_ids,
            grad_fn: Some(std::sync::Arc::new(MulGradFn::<B, D, K>::new(
                self.lhs.clone(),
                self.rhs.clone(),
            ))),
            metadata: HashMap::new(),
        }
    }
}

/// Gradient function for multiplication
#[derive(Debug)]
pub struct MulGradFn<B: Backend, const D: usize, K: DataType> {
    lhs: Tensor<B, D, K>,
    rhs: Tensor<B, D, K>,
}

impl<B: Backend, const D: usize, K: DataType> MulGradFn<B, D, K> {
    pub fn new(lhs: Tensor<B, D, K>, rhs: Tensor<B, D, K>) -> Self {
        Self { lhs, rhs }
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> crate::autodiff::tape::TapeGradFn<B>
    for MulGradFn<B, D, K>
where
    K::Primitive: Clone + std::ops::Mul<Output = K::Primitive> + Send + Sync + 'static,
{
    fn compute_gradients(
        &self,
        grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>> {
        // For multiplication: grad_lhs = grad_output * rhs, grad_rhs = grad_output * lhs
        if let Some(grad) = grad_output.downcast_ref::<Tensor<B, D, K>>() {
            if D == 1 {
                let grad_data = grad.to_data();
                let lhs_data = self.lhs.to_data();
                let rhs_data = self.rhs.to_data();

                let grad_lhs_data: Vec<K::Primitive> = grad_data
                    .iter()
                    .zip(rhs_data.iter())
                    .map(|(g, r)| g.clone() * r.clone())
                    .collect();

                let grad_rhs_data: Vec<K::Primitive> = grad_data
                    .iter()
                    .zip(lhs_data.iter())
                    .map(|(g, l)| g.clone() * l.clone())
                    .collect();

                let grad_lhs: Tensor<B, D, K> =
                    Tensor::from_data(grad_lhs_data, self.lhs.shape().clone());
                let grad_rhs: Tensor<B, D, K> =
                    Tensor::from_data(grad_rhs_data, self.rhs.shape().clone());

                vec![
                    Box::new(grad_lhs) as Box<dyn std::any::Any + Send + Sync>,
                    Box::new(grad_rhs) as Box<dyn std::any::Any + Send + Sync>,
                ]
            } else {
                vec![]
            }
        } else {
            vec![]
        }
    }

    fn description(&self) -> String {
        "MulGradFn".to_string()
    }
}

/// ReLU activation operation
#[derive(Debug, Clone)]
pub struct ReluOp<B: Backend, const D: usize, K: DataType> {
    pub input: Tensor<B, D, K>,
}

impl<B: Backend, const D: usize, K: DataType> ReluOp<B, D, K> {
    pub fn new(input: Tensor<B, D, K>) -> Self {
        Self { input }
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> AutodiffOp<B> for ReluOp<B, D, K>
where
    K::Primitive: Clone + Default + PartialOrd + Send + Sync + 'static,
{
    type Input = ();
    type Output = Tensor<B, D, K>;

    fn forward(&self, _input: Self::Input) -> Self::Output {
        let input_data = self.input.to_data();
        let zero = K::Primitive::default();

        let result_data: Vec<K::Primitive> = input_data
            .iter()
            .map(|x| if *x > zero { x.clone() } else { zero.clone() })
            .collect();

        Tensor::from_data(result_data, self.input.shape().clone())
    }

    fn create_tape_operation(&self, output_id: NodeId, input_ids: Vec<NodeId>) -> TapeOperation<B> {
        TapeOperation {
            op_type: "ReLU".to_string(),
            output_id,
            input_ids,
            grad_fn: Some(std::sync::Arc::new(ReluGradFn::<B, D, K>::new(
                self.input.clone(),
            ))),
            metadata: HashMap::new(),
        }
    }
}

/// Gradient function for ReLU
#[derive(Debug)]
pub struct ReluGradFn<B: Backend, const D: usize, K: DataType> {
    input: Tensor<B, D, K>,
}

impl<B: Backend, const D: usize, K: DataType> ReluGradFn<B, D, K> {
    pub fn new(input: Tensor<B, D, K>) -> Self {
        Self { input }
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> crate::autodiff::tape::TapeGradFn<B>
    for ReluGradFn<B, D, K>
where
    K::Primitive: Clone + Default + PartialOrd + Send + Sync + 'static,
{
    fn compute_gradients(
        &self,
        grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>> {
        // For ReLU: grad_input = grad_output * (input > 0)
        if let Some(grad) = grad_output.downcast_ref::<Tensor<B, D, K>>() {
            let grad_data = grad.to_data();
            let input_data = self.input.to_data();
            let zero = K::Primitive::default();

            let grad_input_data: Vec<K::Primitive> = grad_data
                .iter()
                .zip(input_data.iter())
                .map(|(g, x)| if *x > zero { g.clone() } else { zero.clone() })
                .collect();

            let grad_input: Tensor<B, D, K> =
                Tensor::from_data(grad_input_data, self.input.shape().clone());

            vec![Box::new(grad_input) as Box<dyn std::any::Any + Send + Sync>]
        } else {
            vec![]
        }
    }

    fn description(&self) -> String {
        "ReluGradFn".to_string()
    }
}

/// Sigmoid activation operation
#[derive(Debug, Clone)]
pub struct SigmoidOp<B: Backend, const D: usize> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, const D: usize> SigmoidOp<B, D> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default, const D: usize> AutodiffOp<B> for SigmoidOp<B, D> {
    type Input = Tensor<B, D, crate::Float>;
    type Output = Tensor<B, D, crate::Float>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        let input_data = input.to_data();

        let result_data: Vec<f32> = input_data
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        Tensor::from_data(result_data, input.shape().clone())
    }

    fn create_tape_operation(&self, output_id: NodeId, input_ids: Vec<NodeId>) -> TapeOperation<B> {
        TapeOperation {
            op_type: "Sigmoid".to_string(),
            output_id,
            input_ids,
            grad_fn: Some(std::sync::Arc::new(SigmoidGradFn::<B, D>::new())),
            metadata: HashMap::new(),
        }
    }
}

/// Gradient function for Sigmoid
#[derive(Debug)]
pub struct SigmoidGradFn<B: Backend, const D: usize> {
    _phantom: std::marker::PhantomData<B>,
}

impl<B: Backend, const D: usize> SigmoidGradFn<B, D> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default, const D: usize> crate::autodiff::tape::TapeGradFn<B>
    for SigmoidGradFn<B, D>
{
    fn compute_gradients(
        &self,
        grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>> {
        // For sigmoid: grad_input = grad_output * sigmoid_output * (1 - sigmoid_output)
        // We need to recompute sigmoid since we don't store the forward output
        if let Some(grad) = grad_output.downcast_ref::<Tensor<B, D, crate::Float>>() {
            // This is a simplified implementation that assumes we can access the input
            // In a real implementation, we'd store the sigmoid output during forward pass
            let _grad_data = grad.to_data();

            // For now, just pass through the gradient (placeholder)
            let grad_input = grad.clone();

            vec![Box::new(grad_input) as Box<dyn std::any::Any + Send + Sync>]
        } else {
            vec![]
        }
    }

    fn description(&self) -> String {
        "SigmoidGradFn".to_string()
    }
}

/// Sum reduction operation
#[derive(Debug, Clone)]
pub struct SumOp<B: Backend, const D: usize, K: DataType> {
    pub input_shape: Shape<D>,
    _phantom: std::marker::PhantomData<(B, K)>,
}

impl<B: Backend, const D: usize, K: DataType> SumOp<B, D, K> {
    pub fn new(input_shape: Shape<D>) -> Self {
        Self {
            input_shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> AutodiffOp<B> for SumOp<B, D, K>
where
    K::Primitive: Clone + Default + std::iter::Sum + Send + Sync + 'static,
{
    type Input = Tensor<B, D, K>;
    type Output = Tensor<B, 1, K>;

    fn forward(&self, input: Self::Input) -> Self::Output {
        let input_data = input.to_data();
        let sum_value: K::Primitive = input_data.iter().cloned().sum();

        Tensor::from_data(vec![sum_value], Shape::new([1]))
    }

    fn create_tape_operation(&self, output_id: NodeId, input_ids: Vec<NodeId>) -> TapeOperation<B> {
        let mut metadata = HashMap::new();
        metadata.insert(
            "input_shape".to_string(),
            format!("{:?}", self.input_shape.dims()),
        );

        TapeOperation {
            op_type: "Sum".to_string(),
            output_id,
            input_ids,
            grad_fn: Some(std::sync::Arc::new(SumGradFn::<B, D, K>::new(
                self.input_shape.clone(),
            ))),
            metadata,
        }
    }
}

/// Gradient function for Sum
#[derive(Debug)]
pub struct SumGradFn<B: Backend, const D: usize, K: DataType> {
    input_shape: Shape<D>,
    _phantom: std::marker::PhantomData<(B, K)>,
}

impl<B: Backend, const D: usize, K: DataType> SumGradFn<B, D, K> {
    pub fn new(input_shape: Shape<D>) -> Self {
        Self {
            input_shape,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<B: Backend + Default, const D: usize, K: DataType> crate::autodiff::tape::TapeGradFn<B>
    for SumGradFn<B, D, K>
where
    K::Primitive: Clone + Send + Sync + 'static,
{
    fn compute_gradients(
        &self,
        grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>> {
        // For sum: grad_input = grad_output broadcasted to input shape
        if let Some(grad) = grad_output.downcast_ref::<Tensor<B, 1, K>>() {
            let grad_data = grad.to_data();
            let grad_scalar = grad_data[0].clone();

            // Broadcast the scalar gradient to the input shape
            let input_size = self.input_shape.num_elements();
            let grad_input_data = vec![grad_scalar; input_size];

            let grad_input: Tensor<B, D, K> =
                Tensor::from_data(grad_input_data, self.input_shape.clone());

            vec![Box::new(grad_input) as Box<dyn std::any::Any + Send + Sync>]
        } else {
            vec![]
        }
    }

    fn description(&self) -> String {
        "SumGradFn".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autodiff::tape::TapeGradFn;
    use crate::{CpuBackend, Shape, Tensor};

    #[test]
    fn test_add_op_forward() {
        let shape = Shape::new([3]);
        let lhs = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0, 3.0], shape.clone());
        let rhs = Tensor::<CpuBackend, 1>::from_data(vec![4.0, 5.0, 6.0], shape.clone());

        let add_op = AddOp::<CpuBackend, 1, crate::Float>::new(shape.clone(), shape.clone());
        let result = add_op.forward((lhs, rhs));

        let result_data = result.to_data();
        assert_eq!(result_data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_relu_op_forward() {
        let shape = Shape::new([4]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![-1.0, 0.0, 1.0, 2.0], shape.clone());

        let relu_op = ReluOp::<CpuBackend, 1, crate::Float>::new(input);
        let result = relu_op.forward(());

        let result_data = result.to_data();
        assert_eq!(result_data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sum_op_forward() {
        let shape = Shape::new([3]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0, 3.0], shape.clone());

        let sum_op = SumOp::<CpuBackend, 1, crate::Float>::new(shape);
        let result = sum_op.forward(input);

        let result_data = result.to_data();
        assert_eq!(result_data, vec![6.0]);
    }

    #[test]
    fn test_sigmoid_op_forward() {
        let shape = Shape::new([2]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![0.0, 1.0], shape.clone());

        let sigmoid_op = SigmoidOp::<CpuBackend, 1>::new();
        let result = sigmoid_op.forward(input);

        let result_data = result.to_data();
        assert!((result_data[0] - 0.5).abs() < 1e-6);
        assert!((result_data[1] - 0.7310586).abs() < 1e-6);
    }

    #[test]
    fn test_add_grad_fn() {
        let shape = Shape::new([2]);
        let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0], shape.clone());

        let add_grad_fn =
            AddGradFn::<CpuBackend, 1, crate::Float>::new(shape.clone(), shape.clone());
        let gradients = add_grad_fn.compute_gradients(&grad_output);

        assert_eq!(gradients.len(), 2);

        let grad_lhs = gradients[0]
            .downcast_ref::<Tensor<CpuBackend, 1>>()
            .unwrap();
        let grad_rhs = gradients[1]
            .downcast_ref::<Tensor<CpuBackend, 1>>()
            .unwrap();

        assert_eq!(grad_lhs.to_data(), vec![1.0, 1.0]);
        assert_eq!(grad_rhs.to_data(), vec![1.0, 1.0]);
    }

    #[test]
    fn test_relu_grad_fn() {
        let shape = Shape::new([3]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![-1.0, 0.0, 1.0], shape.clone());
        let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0, 1.0], shape.clone());

        let relu_grad_fn = ReluGradFn::<CpuBackend, 1, crate::Float>::new(input);
        let gradients = relu_grad_fn.compute_gradients(&grad_output);

        assert_eq!(gradients.len(), 1);

        let grad_input = gradients[0]
            .downcast_ref::<Tensor<CpuBackend, 1>>()
            .unwrap();
        assert_eq!(grad_input.to_data(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sum_grad_fn() {
        let shape = Shape::new([3]);
        let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![2.0], Shape::new([1]));

        let sum_grad_fn = SumGradFn::<CpuBackend, 1, crate::Float>::new(shape);
        let gradients = sum_grad_fn.compute_gradients(&grad_output);

        assert_eq!(gradients.len(), 1);

        let grad_input = gradients[0]
            .downcast_ref::<Tensor<CpuBackend, 3>>()
            .unwrap();
        assert_eq!(grad_input.to_data(), vec![2.0, 2.0, 2.0]);
    }
}
