//! Simplified Autodiff Demo
//!
//! This example demonstrates the core automatic differentiation capabilities
//! of Mini-Burn with basic operations and gradient computation.

use mini_burn::autodiff::ops::{AddOp, AutodiffOp, MulOp, ReluOp, SumOp};
use mini_burn::autodiff::tape::TapeGradFn;
use mini_burn::autodiff::test_utils::NumericalGradChecker;
use mini_burn::{CpuBackend, Shape, Tensor};

fn main() {
    println!("=== Mini-Burn Autodiff Demo ===\n");

    // Basic arithmetic operations
    demo_basic_arithmetic();

    // Activation functions
    demo_activation_functions();

    // Reduction operations
    demo_reduction_operations();

    // Numerical gradient checking
    demo_numerical_gradient_checking();

    println!("🎉 All autodiff demos completed successfully!");
}

fn demo_basic_arithmetic() {
    println!("📊 Demo: Basic Arithmetic Operations");
    println!("====================================");

    // Create test tensors
    let shape = Shape::new([3]);
    let a = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0, 3.0], shape.clone());
    let b = Tensor::<CpuBackend, 1>::from_data(vec![4.0, 5.0, 6.0], shape.clone());

    println!("Input A: {:?}", a.to_data());
    println!("Input B: {:?}", b.to_data());

    // Test addition
    {
        let add_op = AddOp::<CpuBackend, 1, mini_burn::Float>::new(shape.clone(), shape.clone());
        let result = add_op.forward((a.clone(), b.clone()));
        println!("A + B = {:?}", result.to_data());

        // Test gradient computation
        let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0, 1.0], shape.clone());
        let add_grad_fn: mini_burn::autodiff::AddGradFn<CpuBackend, 1, mini_burn::Float> =
            mini_burn::autodiff::ops::AddGradFn::new(shape.clone(), shape.clone());
        let gradients = add_grad_fn.compute_gradients(&grad_output);

        if gradients.len() == 2 {
            if let (Some(grad_a), Some(grad_b)) = (
                gradients[0].downcast_ref::<Tensor<CpuBackend, 1>>(),
                gradients[1].downcast_ref::<Tensor<CpuBackend, 1>>(),
            ) {
                println!("Grad A: {:?}", grad_a.to_data());
                println!("Grad B: {:?}", grad_b.to_data());
            }
        }
    }

    // Test multiplication
    {
        let mul_op = MulOp::<CpuBackend, 1, mini_burn::Float>::new(a.clone(), b.clone());
        let result = mul_op.forward(());
        println!("A * B = {:?}", result.to_data());

        // Test gradient computation
        let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0, 1.0], shape.clone());
        let mul_grad_fn = mini_burn::autodiff::ops::MulGradFn::new(a.clone(), b.clone());
        let gradients = mul_grad_fn.compute_gradients(&grad_output);

        if gradients.len() == 2 {
            if let (Some(grad_a), Some(grad_b)) = (
                gradients[0].downcast_ref::<Tensor<CpuBackend, 1>>(),
                gradients[1].downcast_ref::<Tensor<CpuBackend, 1>>(),
            ) {
                println!("Grad A (A*B): {:?}", grad_a.to_data());
                println!("Grad B (A*B): {:?}", grad_b.to_data());
            }
        }
    }

    println!();
}

fn demo_activation_functions() {
    println!("🔥 Demo: Activation Functions");
    println!("=============================");

    let shape = Shape::new([4]);
    let input = Tensor::<CpuBackend, 1>::from_data(vec![-2.0, -1.0, 1.0, 2.0], shape.clone());

    println!("Input: {:?}", input.to_data());

    // Test ReLU
    {
        let relu_op = ReluOp::<CpuBackend, 1, mini_burn::Float>::new(input.clone());
        let result = relu_op.forward(());
        println!("ReLU(input) = {:?}", result.to_data());

        // Test ReLU gradient
        let grad_output =
            Tensor::<CpuBackend, 1>::from_data(vec![1.0, 1.0, 1.0, 1.0], shape.clone());
        let relu_grad_fn = mini_burn::autodiff::ops::ReluGradFn::new(input.clone());
        let gradients = relu_grad_fn.compute_gradients(&grad_output);

        if !gradients.is_empty() {
            if let Some(grad_input) = gradients[0].downcast_ref::<Tensor<CpuBackend, 1>>() {
                println!("ReLU gradient: {:?}", grad_input.to_data());
            }
        }
    }

    // Test Sigmoid
    {
        let sigmoid_op = mini_burn::autodiff::ops::SigmoidOp::<CpuBackend, 1>::new();
        let result = sigmoid_op.forward(input.clone());
        println!("Sigmoid(input) = {:?}", result.to_data());
    }

    println!();
}

fn demo_reduction_operations() {
    println!("📈 Demo: Reduction Operations");
    println!("=============================");

    let shape = Shape::new([4]);
    let input = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());

    println!("Input: {:?}", input.to_data());

    // Test Sum
    {
        let sum_op = SumOp::<CpuBackend, 1, mini_burn::Float>::new(shape.clone());
        let result = sum_op.forward(input.clone());
        println!("Sum(input) = {:?}", result.to_data());

        // Test Sum gradient
        let grad_output = Tensor::<CpuBackend, 1>::from_data(vec![1.0], Shape::new([1]));
        let sum_grad_fn: mini_burn::autodiff::SumGradFn<CpuBackend, 1, mini_burn::Float> =
            mini_burn::autodiff::ops::SumGradFn::new(shape.clone());
        let gradients = sum_grad_fn.compute_gradients(&grad_output);

        if !gradients.is_empty() {
            if let Some(grad_input) = gradients[0].downcast_ref::<Tensor<CpuBackend, 1>>() {
                println!("Sum gradient: {:?}", grad_input.to_data());
            }
        }
    }

    println!();
}

fn demo_numerical_gradient_checking() {
    println!("🧮 Demo: Numerical Gradient Checking");
    println!("====================================");

    let checker = NumericalGradChecker::default();

    // Test L2 norm squared: f(x) = ||x||^2
    {
        println!("Testing L2 norm squared gradient...");
        let shape = Shape::new([3]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![2.0, 3.0, 4.0], shape.clone());

        // Analytical gradient for ||x||^2 is 2*x
        let analytical_grad =
            Tensor::<CpuBackend, 1>::from_data(vec![4.0, 6.0, 8.0], shape.clone());

        let inputs = vec![input];
        let analytical_grads = vec![analytical_grad];

        match checker.check_gradients(
            &inputs,
            mini_burn::autodiff::test_utils::l2_norm_squared,
            &analytical_grads,
        ) {
            Ok(()) => println!("✅ L2 norm squared gradient check passed!"),
            Err(e) => println!("❌ L2 norm squared gradient check failed: {}", e),
        }
    }

    // Test dot product: f(x, y) = x • y
    {
        println!("Testing dot product gradient...");
        let shape = Shape::new([2]);
        let x = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0], shape.clone());
        let y = Tensor::<CpuBackend, 1>::from_data(vec![3.0, 4.0], shape.clone());

        // Analytical gradients for dot product: df/dx = y, df/dy = x
        let grad_x = y.clone();
        let grad_y = x.clone();

        let inputs = vec![x, y];
        let analytical_grads = vec![grad_x, grad_y];

        match checker.check_gradients(
            &inputs,
            mini_burn::autodiff::test_utils::dot_product,
            &analytical_grads,
        ) {
            Ok(()) => println!("✅ Dot product gradient check passed!"),
            Err(e) => println!("❌ Dot product gradient check failed: {}", e),
        }
    }

    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let shape = Shape::new([2]);
        let a = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0], shape.clone());
        let b = Tensor::<CpuBackend, 1>::from_data(vec![3.0, 4.0], shape.clone());

        // Test addition
        let add_op = AddOp::<CpuBackend, 1, mini_burn::Float>::new(shape.clone(), shape.clone());
        let result = add_op.forward((a.clone(), b.clone()));
        assert_eq!(result.to_data(), vec![4.0, 6.0]);

        // Test multiplication
        let mul_op = MulOp::<CpuBackend, 1, mini_burn::Float>::new(a.clone(), b.clone());
        let result = mul_op.forward(());
        assert_eq!(result.to_data(), vec![3.0, 8.0]);
    }

    #[test]
    fn test_relu_activation() {
        let shape = Shape::new([4]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![-1.0, 0.0, 1.0, 2.0], shape.clone());

        let relu_op = ReluOp::<CpuBackend, 1, mini_burn::Float>::new(input);
        let result = relu_op.forward(());
        assert_eq!(result.to_data(), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_sum_operation() {
        let shape = Shape::new([3]);
        let input = Tensor::<CpuBackend, 1>::from_data(vec![1.0, 2.0, 3.0], shape.clone());

        let sum_op = SumOp::<CpuBackend, 1, mini_burn::Float>::new(shape);
        let result = sum_op.forward(input);
        assert_eq!(result.to_data(), vec![6.0]);
    }
}
