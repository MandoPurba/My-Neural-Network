//! Advanced operations and use cases for mini-burn framework

use mini_burn::{CpuBackend, Int, Shape, Tensor};

fn main() {
    println!("=== Advanced Mini-Burn Operations ===\n");

    // 1. Multi-dimensional tensors
    println!("1. Multi-dimensional Tensors:");

    // 1D tensor (vector)
    let vector: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], Shape::new([5]));
    println!("1D Vector: {:?}", vector.to_data());

    // 2D tensor (matrix)
    let matrix: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3]));
    println!("2D Matrix (2x3): {:?}", matrix.to_data());

    // 3D tensor (cube)
    let cube: Tensor<CpuBackend, 3> = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        Shape::new([2, 2, 2]),
    );
    println!("3D Cube (2x2x2): {:?}", cube.to_data());

    // 4D tensor (batch of cubes - common in deep learning)
    let batch: Tensor<CpuBackend, 4> = Tensor::from_data(
        (1..=16).map(|x| x as f32).collect(),
        Shape::new([2, 2, 2, 2]),
    );
    println!("4D Batch (2x2x2x2): {:?}\n", batch.to_data());

    // 2. Chain operations
    println!("2. Chained Operations:");
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.5, 1.0, 1.5, 2.0], Shape::new([2, 2]));

    println!("A: {:?}", a.to_data());
    println!("B: {:?}", b.to_data());

    // (A + B) * A
    let result = a.add(&b).mul(&a);
    println!("(A + B) * A: {:?}", result.to_data());

    // A + scalar, then multiply by B
    let result2 = a.add_scalar(1.0).mul(&b);
    println!("(A + 1) * B: {:?}\n", result2.to_data());

    // 3. Matrix operations
    println!("3. Matrix Operations:");

    // Create matrices for multiplication
    let mat1: Tensor<CpuBackend, 2> = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        Shape::new([2, 3]), // 2x3 matrix
    );
    let mat2: Tensor<CpuBackend, 2> = Tensor::from_data(
        vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        Shape::new([3, 2]), // 3x2 matrix
    );

    println!("Matrix 1 (2x3): {:?}", mat1.to_data());
    println!("Matrix 2 (3x2): {:?}", mat2.to_data());

    let mat_result = mat1.matmul(&mat2);
    println!("Matrix1 @ Matrix2 (2x2): {:?}", mat_result.to_data());
    println!("Expected: [58, 64, 139, 154]\n");

    // 4. Mathematical functions showcase
    println!("4. Mathematical Functions:");
    let math_input: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![0.0, 0.5, 1.0, 1.5, 2.0], Shape::new([5]));

    println!("Input: {:?}", math_input.to_data());
    println!("Sin: {:?}", math_input.sin().to_data());
    println!("Cos: {:?}", math_input.cos().to_data());
    println!("Exp: {:?}", math_input.exp().to_data());
    println!("Sqrt: {:?}", math_input.sqrt().to_data());
    println!("Square (pow 2): {:?}", math_input.pow(2.0).to_data());
    println!("Abs: {:?}", math_input.sub_scalar(1.0).abs().to_data());

    // 5. Complex mathematical chains
    println!("\n5. Complex Mathematical Chains:");
    let x: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));

    // sigmoid-like function: 1 / (1 + exp(-x))
    let sigmoid = Tensor::ones(Shape::new([2, 2]))
        .div(&Tensor::ones(Shape::new([2, 2])).add(&x.mul_scalar(-1.0).exp()));

    println!("X: {:?}", x.to_data());
    println!("Sigmoid(X): {:?}", sigmoid.to_data());

    // ReLU-like function: max(0, x)
    let relu_input: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![-2.0, -1.0, 1.0, 2.0], Shape::new([2, 2]));
    let zeros = Tensor::<CpuBackend, 2>::zeros(Shape::new([2, 2]));

    println!("ReLU input: {:?}", relu_input.to_data());
    // Simple ReLU approximation using sigmoid
    let relu_approx = relu_input.clone().mul(
        &relu_input
            .clone()
            .mul_scalar(10.0)
            .exp()
            .div(&Tensor::ones(Shape::new([2, 2])).add(&relu_input.mul_scalar(10.0).exp())),
    );
    println!("ReLU approximation: {:?}", relu_approx.to_data());

    // 6. Different data types in action
    println!("\n6. Different Data Types:");

    // Integer computations
    let int_a: Tensor<CpuBackend, 2, Int> =
        Tensor::from_data(vec![10, 20, 30, 40], Shape::new([2, 2]));
    let int_b: Tensor<CpuBackend, 2, Int> = Tensor::from_data(vec![1, 2, 3, 4], Shape::new([2, 2]));

    println!("Int A: {:?}", int_a.to_data());
    println!("Int B: {:?}", int_b.to_data());
    println!("Note: Int operations would require implementing ops trait for Int type");

    // 7. Large tensor operations
    println!("\n7. Large Tensor Operations:");

    let large_shape = Shape::new([10, 10]);
    let large_a = Tensor::<CpuBackend, 2>::ones(large_shape.clone());
    let large_b = Tensor::<CpuBackend, 2>::full(large_shape, 2.0);

    println!("Large tensor A (10x10) filled with 1.0");
    println!("Large tensor B (10x10) filled with 2.0");

    let large_result = large_a.add(&large_b);
    println!("A + B should all be 3.0");
    println!("First few elements: {:?}", &large_result.to_data()[..5]);
    println!("Total elements: {}", large_result.numel());

    // Matrix multiplication with larger matrices
    let large_mat_result = large_a.matmul(&large_b);
    println!("Large matrix multiplication result (10x10):");
    println!("First few elements: {:?}", &large_mat_result.to_data()[..5]);
    println!("Expected each element to be 20.0 (10 * 1.0 * 2.0)");

    // 8. Statistical operations
    println!("\n8. Statistical Operations:");
    let stats_tensor: Tensor<CpuBackend, 2> = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        Shape::new([3, 3]),
    );

    println!("Stats tensor (3x3): {:?}", stats_tensor.to_data());
    println!("Sum: {}", stats_tensor.sum());
    println!("Mean: {}", stats_tensor.mean());
    println!("Max: {}", stats_tensor.max());
    println!("Min: {}", stats_tensor.min());

    // Variance approximation: mean((x - mean)^2)
    let mean_val = stats_tensor.mean();
    let variance_approx = stats_tensor.sub_scalar(mean_val).pow(2.0).mean();
    println!("Variance (approximation): {}", variance_approx);

    println!("\n=== Advanced Operations Complete ===");
}
