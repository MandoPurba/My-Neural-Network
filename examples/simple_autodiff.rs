//! Simple Autodiff Demo
//!
//! This example demonstrates basic automatic differentiation
//! functionality without complex dependencies.

use mini_burn::autodiff::ops::{
    AddGradFn, AddOp, AutodiffOp, MulGradFn, MulOp, ReluGradFn, ReluOp, SumGradFn, SumOp,
};
use mini_burn::autodiff::tape::TapeGradFn;
use mini_burn::{CpuBackend, Shape, Tensor};

fn main() {
    println!("=== Simple Mini-Burn Autodiff Demo ===\n");

    // Test basic operations
    test_addition();
    test_multiplication();
    test_relu();
    test_sum();

    println!("🎉 Simple autodiff demo completed!");
}

fn test_addition() {
    println!("➕ Testing Addition Operation");
    println!("-----------------------------");

    let shape = Shape::new([3]);
    let a = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0, 3.0], shape.clone());
    let b = Tensor::<CpuBackend, 1>::from_data(vec![4.0, 5.0, 6.0], shape.clone());

    println!("Input A: {:?}", a.to_data());
    println!("Input B: {:?}", b.to_data());

    let add_op = AddOp::<CpuBackend, 1, mini_burn::Float>::new(shape.clone(), shape.clone());
    let result = add_op.forward((a.clone(), b.clone()));
    println!("A + B = {:?}", result.to_data());

    // Test gradient computation
    let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0, 1.0], shape.clone());
    let add_grad_fn =
        AddGradFn::<CpuBackend, 1, mini_burn::Float>::new(shape.clone(), shape.clone());
    let gradients = add_grad_fn.compute_gradients(&grad_output);

    if gradients.len() == 2 {
        if let (Some(grad_a), Some(grad_b)) = (
            gradients[0].downcast_ref::<Tensor<CpuBackend, 1>>(),
            gradients[1].downcast_ref::<Tensor<CpuBackend, 1>>(),
        ) {
            println!("Gradient w.r.t. A: {:?}", grad_a.to_data());
            println!("Gradient w.r.t. B: {:?}", grad_b.to_data());
        }
    }
    println!();
}

fn test_multiplication() {
    println!("✖️ Testing Multiplication Operation");
    println!("----------------------------------");

    let shape = Shape::new([2]);
    let a = Tensor::<CpuBackend, 1>::from_data(vec![2.0, 3.0], shape.clone());
    let b = Tensor::<CpuBackend, 1>::from_data(vec![4.0, 5.0], shape.clone());

    println!("Input A: {:?}", a.to_data());
    println!("Input B: {:?}", b.to_data());

    let mul_op = MulOp::<CpuBackend, 1, mini_burn::Float>::new(a.clone(), b.clone());
    let result = mul_op.forward(());
    println!("A * B = {:?}", result.to_data());

    // Test gradient computation
    let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0], shape.clone());
    let mul_grad_fn = MulGradFn::<CpuBackend, 1, mini_burn::Float>::new(a.clone(), b.clone());
    let gradients = mul_grad_fn.compute_gradients(&grad_output);

    if gradients.len() == 2 {
        if let (Some(grad_a), Some(grad_b)) = (
            gradients[0].downcast_ref::<Tensor<CpuBackend, 1>>(),
            gradients[1].downcast_ref::<Tensor<CpuBackend, 1>>(),
        ) {
            println!("Gradient w.r.t. A: {:?}", grad_a.to_data());
            println!("Gradient w.r.t. B: {:?}", grad_b.to_data());
        }
    }
    println!();
}

fn test_relu() {
    println!("🔥 Testing ReLU Activation");
    println!("---------------------------");

    let shape = Shape::new([4]);
    let input = Tensor::<CpuBackend, 1>::from_data(vec![-2.0, -1.0, 1.0, 2.0], shape.clone());

    println!("Input: {:?}", input.to_data());

    let relu_op = ReluOp::<CpuBackend, 1, mini_burn::Float>::new(input.clone());
    let result = relu_op.forward(());
    println!("ReLU(input) = {:?}", result.to_data());

    // Test ReLU gradient
    let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0, 1.0, 1.0], shape.clone());
    let relu_grad_fn = ReluGradFn::<CpuBackend, 1, mini_burn::Float>::new(input.clone());
    let gradients = relu_grad_fn.compute_gradients(&grad_output);

    if !gradients.is_empty() {
        if let Some(grad_input) = gradients[0].downcast_ref::<Tensor<CpuBackend, 1>>() {
            println!("ReLU gradient: {:?}", grad_input.to_data());
        }
    }
    println!();
}

fn test_sum() {
    println!("📈 Testing Sum Reduction");
    println!("-------------------------");

    let shape = Shape::new([4]);
    let input = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());

    println!("Input: {:?}", input.to_data());

    let sum_op = SumOp::<CpuBackend, 1, mini_burn::Float>::new(shape.clone());
    let result = sum_op.forward(input.clone());
    println!("Sum(input) = {:?}", result.to_data());

    // Test Sum gradient
    let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0], Shape::new([1]));
    let sum_grad_fn = SumGradFn::<CpuBackend, 1, mini_burn::Float>::new(shape.clone());
    let gradients = sum_grad_fn.compute_gradients(&grad_output);

    if !gradients.is_empty() {
        if let Some(grad_input) = gradients[0].downcast_ref::<Tensor<CpuBackend, 1>>() {
            println!("Sum gradient: {:?}", grad_input.to_data());
        }
    }
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_addition() {
        let shape = Shape::new([2]);
        let a = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0], shape.clone());
        let b = Tensor::<CpuBackend, 1>::from_data(vec![3.0, 4.0], shape.clone());

        let add_op = AddOp::<CpuBackend, 1, mini_burn::Float>::new(shape.clone(), shape.clone());
        let result = add_op.forward((a, b));
        assert_eq!(result.to_data(), vec![4.0, 6.0]);
    }

    #[test]
    fn test_simple_relu() {
        let shape = Shape::new([3]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![-1.0, 0.0, 1.0], shape.clone());

        let relu_op = ReluOp::<CpuBackend, 1, mini_burn::Float>::new(input);
        let result = relu_op.forward(());
        assert_eq!(result.to_data(), vec![0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_gradient_shapes() {
        let shape = Shape::new([2]);
        let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0], shape.clone());

        let add_grad_fn =
            AddGradFn::<CpuBackend, 1, mini_burn::Float>::new(shape.clone(), shape.clone());
        let gradients = add_grad_fn.compute_gradients(&grad_output);

        assert_eq!(gradients.len(), 2);
    }
}
