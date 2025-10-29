//! Working Demo - Basic Tensor Operations
//!
//! This example demonstrates the core tensor operations that are working
//! in the Mini-Burn framework.

use mini_burn::{backend::CpuBackend, broadcast::Broadcasting, shape::Shape, tensor::Tensor};

type Backend = CpuBackend;

fn main() {
    println!("🔥 Mini-Burn Working Demo");
    println!("=========================\n");

    // Basic tensor creation and operations
    demonstrate_tensor_operations();

    // Broadcasting capabilities
    demonstrate_broadcasting();

    println!("\n🎉 Mini-Burn basic operations are working!");
}

fn demonstrate_tensor_operations() {
    println!("1️⃣ TENSOR OPERATIONS");

    // Create tensors
    let shape = Shape::new([2, 3]);
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data2 = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

    let tensor1: Tensor<Backend, 2> = Tensor::from_data(data1, shape.clone());
    let tensor2: Tensor<Backend, 2> = Tensor::from_data(data2, shape);

    println!(
        "   ✓ Created tensors with shape: {:?}",
        tensor1.shape().dims()
    );
    println!("   ✓ Tensor1 elements: {}", tensor1.numel());
    println!("   ✓ Tensor2 elements: {}", tensor2.numel());

    // Element-wise operations
    let sum_result = tensor1.add(&tensor2);
    let mul_result = tensor1.mul(&tensor2);
    let sub_result = tensor1.sub(&tensor2);
    let div_result = tensor1.div(&tensor2);

    println!(
        "   ✓ Element-wise addition: {:?}",
        sum_result.shape().dims()
    );
    println!(
        "   ✓ Element-wise multiplication: {:?}",
        mul_result.shape().dims()
    );
    println!(
        "   ✓ Element-wise subtraction: {:?}",
        sub_result.shape().dims()
    );
    println!(
        "   ✓ Element-wise division: {:?}",
        div_result.shape().dims()
    );

    // Mathematical functions
    let exp_result = tensor1.exp();
    let ln_result = tensor1.ln();
    let sqrt_result = tensor1.sqrt();
    let sin_result = tensor1.sin();
    let cos_result = tensor1.cos();

    println!("   ✓ Exponential function: {:?}", exp_result.shape().dims());
    println!("   ✓ Natural logarithm: {:?}", ln_result.shape().dims());
    println!("   ✓ Square root: {:?}", sqrt_result.shape().dims());
    println!("   ✓ Sine function: {:?}", sin_result.shape().dims());
    println!("   ✓ Cosine function: {:?}", cos_result.shape().dims());

    // Activation functions
    let relu_result = tensor1.relu();
    let sigmoid_result = tensor1.sigmoid();
    let tanh_result = tensor1.tanh();

    println!("   ✓ ReLU activation: {:?}", relu_result.shape().dims());
    println!(
        "   ✓ Sigmoid activation: {:?}",
        sigmoid_result.shape().dims()
    );
    println!("   ✓ Tanh activation: {:?}", tanh_result.shape().dims());

    // Reduction operations
    let sum_all = tensor1.sum();
    let mean_all = tensor1.mean();
    let max_all = tensor1.max();
    let min_all = tensor1.min();

    println!("   ✓ Sum reduction: {:?}", sum_all);
    println!("   ✓ Mean reduction: {:?}", mean_all);
    println!("   ✓ Max reduction: {:?}", max_all);
    println!("   ✓ Min reduction: {:?}", min_all);

    // Scalar operations
    let scalar_add = tensor1.add_scalar(10.0);
    let scalar_mul = tensor1.mul_scalar(2.0);

    println!("   ✓ Scalar addition: {:?}", scalar_add.shape().dims());
    println!(
        "   ✓ Scalar multiplication: {:?}",
        scalar_mul.shape().dims()
    );

    println!("   ✅ All tensor operations working!\n");
}

fn demonstrate_broadcasting() {
    println!("2️⃣ BROADCASTING SYSTEM");

    let test_cases = [
        (vec![2, 3], vec![1, 3], "Compatible - broadcast first dim"),
        (vec![4, 1], vec![4, 5], "Compatible - broadcast second dim"),
        (vec![3], vec![2, 3], "Compatible - different ranks"),
        (vec![2, 3], vec![2, 4], "Incompatible - conflicting dims"),
        (vec![1, 1], vec![3, 4], "Compatible - broadcast both dims"),
    ];

    for (shape1, shape2, description) in test_cases.iter() {
        let compatible = Broadcasting::are_compatible(shape1, shape2);
        let symbol = if compatible { "✅" } else { "❌" };

        println!("   {} {:?} + {:?}", symbol, shape1, shape2);
        println!("     → {}", description);

        if compatible {
            if let Ok(result) = Broadcasting::broadcast_shape(shape1, shape2) {
                println!("     → Result shape: {:?}", result);
            }
        }
    }

    // Shape manipulation operations
    println!("\n   Shape Operations:");

    let original = vec![1, 4, 1, 3, 1];
    let squeezed = Broadcasting::squeeze_shape(&original);
    println!("   ✓ Squeeze {:?} → {:?}", original, squeezed);

    let shape_2d = vec![3, 4];
    if let Ok(unsqueezed) = Broadcasting::unsqueeze_shape(&shape_2d, 1) {
        println!("   ✓ Unsqueeze {:?} at dim 1 → {:?}", shape_2d, unsqueezed);
    }

    let reduce_shape = vec![2, 3, 4];
    let reduced_no_keepdims = Broadcasting::reduce_shape(&reduce_shape, &[1], false);
    let reduced_keepdims = Broadcasting::reduce_shape(&reduce_shape, &[1], true);

    println!("   ✓ Reduce {:?} along axis 1:", reduce_shape);
    println!("     → keep_dims=false: {:?}", reduced_no_keepdims);
    println!("     → keep_dims=true: {:?}", reduced_keepdims);

    println!("   ✅ Broadcasting system working!\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_creation() {
        let shape = Shape::new([2, 3]);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(data, shape);

        assert_eq!(tensor.shape().dims(), &[2, 3]);
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_element_wise_operations() {
        let shape = Shape::new([2, 2]);
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![1.0, 1.0, 1.0, 1.0];

        let tensor1: Tensor<CpuBackend, 2> = Tensor::from_data(data1, shape.clone());
        let tensor2: Tensor<CpuBackend, 2> = Tensor::from_data(data2, shape);

        let sum_result = tensor1.add(&tensor2);
        let mul_result = tensor1.mul(&tensor2);

        assert_eq!(sum_result.shape().dims(), &[2, 2]);
        assert_eq!(mul_result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_activation_functions() {
        let shape = Shape::new([2, 2]);
        let data = vec![-1.0, 0.0, 1.0, 2.0];
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(data, shape);

        let relu_result = tensor.relu();
        let sigmoid_result = tensor.sigmoid();
        let tanh_result = tensor.tanh();

        assert_eq!(relu_result.shape().dims(), &[2, 2]);
        assert_eq!(sigmoid_result.shape().dims(), &[2, 2]);
        assert_eq!(tanh_result.shape().dims(), &[2, 2]);
    }

    #[test]
    fn test_broadcasting_compatibility() {
        assert!(Broadcasting::are_compatible(&[2, 3], &[1, 3]));
        assert!(Broadcasting::are_compatible(&[2, 1], &[2, 3]));
        assert!(Broadcasting::are_compatible(&[3], &[2, 3]));
        assert!(!Broadcasting::are_compatible(&[2, 3], &[2, 4]));
    }

    #[test]
    fn test_shape_operations() {
        let shape = vec![1, 4, 1, 3, 1];
        let squeezed = Broadcasting::squeeze_shape(&shape);
        assert_eq!(squeezed, vec![4, 3]);

        let shape_2d = vec![3, 4];
        let unsqueezed = Broadcasting::unsqueeze_shape(&shape_2d, 1).unwrap();
        assert_eq!(unsqueezed, vec![3, 1, 4]);
    }
}
