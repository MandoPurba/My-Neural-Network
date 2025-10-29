//! Basic usage examples for mini-burn framework

use mini_burn::{Bool, CpuBackend, Float, Int, Shape, Tensor};

fn main() {
    println!("=== Mini-Burn Framework Demo ===\n");

    // 1. Basic tensor creation
    println!("1. Basic Tensor Creation:");
    let shape = Shape::new([2, 3]);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(data, shape);

    println!("Float tensor: {:?}", tensor);
    println!("Shape: {}", tensor.shape());
    println!("Data: {:?}\n", tensor.to_data());

    // 2. Different data types
    println!("2. Different Data Types:");
    let shape_2x2 = Shape::new([2, 2]);

    // Default float tensor
    let float_tensor: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.5, 2.5, 3.5, 4.5], shape_2x2.clone());
    println!("Float tensor: {:?}", float_tensor);

    // Explicit float tensor
    let explicit_float: Tensor<CpuBackend, 2, Float> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape_2x2.clone());
    println!("Explicit float: {:?}", explicit_float);

    // Int tensor
    let int_tensor: Tensor<CpuBackend, 2, Int> =
        Tensor::from_data(vec![10, 20, 30, 40], shape_2x2.clone());
    println!("Int tensor: {:?}", int_tensor);

    // Bool tensor
    let bool_tensor: Tensor<CpuBackend, 2, Bool> =
        Tensor::from_data(vec![true, false, true, false], shape_2x2);
    println!("Bool tensor: {:?}\n", bool_tensor);

    // 3. Tensor factory methods
    println!("3. Tensor Factory Methods:");
    let zeros: Tensor<CpuBackend, 2> = Tensor::zeros(Shape::new([3, 3]));
    println!("Zeros tensor: {:?}", zeros.to_data());

    let ones: Tensor<CpuBackend, 2> = Tensor::ones(Shape::new([2, 4]));
    println!("Ones tensor: {:?}", ones.to_data());

    let filled: Tensor<CpuBackend, 2> = Tensor::full(Shape::new([2, 2]), 42.0);
    println!("Filled tensor (42.0): {:?}", filled.to_data());

    let range: Tensor<CpuBackend, 1> = Tensor::range(0.0, 10.0, 2.0);
    println!("Range tensor [0..10 step 2]: {:?}\n", range.to_data());

    // 4. Basic operations
    println!("4. Basic Operations:");
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2]));

    println!("Tensor A: {:?}", a.to_data());
    println!("Tensor B: {:?}", b.to_data());

    let sum = a.add(&b);
    println!("A + B = {:?}", sum.to_data());

    let diff = a.sub(&b);
    println!("A - B = {:?}", diff.to_data());

    let prod = a.mul(&b);
    println!("A * B (element-wise) = {:?}", prod.to_data());

    let scalar_add = a.add_scalar(10.0);
    println!("A + 10 = {:?}", scalar_add.to_data());

    // 5. Matrix multiplication
    println!("\n5. Matrix Multiplication:");
    let mat_a: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let mat_b: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2]));

    println!("Matrix A: {:?}", mat_a.to_data());
    println!("Matrix B: {:?}", mat_b.to_data());

    let matmul_result = mat_a.matmul(&mat_b);
    println!("A @ B = {:?}", matmul_result.to_data());
    println!("Expected: [19, 22, 43, 50] (matrix multiplication)");

    // 6. Advanced mathematical operations
    println!("\n6. Advanced Mathematical Operations:");
    let math_tensor: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![0.0, 1.0, 2.0, 3.0], Shape::new([2, 2]));

    println!("Original: {:?}", math_tensor.to_data());
    println!("Sin: {:?}", math_tensor.sin().to_data());
    println!("Cos: {:?}", math_tensor.cos().to_data());
    println!("Exp: {:?}", math_tensor.exp().to_data());

    // 7. Reduction operations
    println!("\n7. Reduction Operations:");
    let reduce_tensor: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));

    println!("Tensor: {:?}", reduce_tensor.to_data());
    println!("Sum: {}", reduce_tensor.sum());
    println!("Mean: {}", reduce_tensor.mean());
    println!("Max: {}", reduce_tensor.max());
    println!("Min: {}", reduce_tensor.min());

    println!("\n=== Demo Complete ===");
}
