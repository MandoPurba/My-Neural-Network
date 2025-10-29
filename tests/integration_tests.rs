//! Integration tests for mini-burn framework

use mini_burn::{Bool, CpuBackend, Float, Int, Shape, Tensor};

#[test]
fn test_tensor_creation_and_basic_properties() {
    let shape = Shape::new([2, 3]);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(data.clone(), shape);

    assert_eq!(tensor.shape().dims(), &[2, 3]);
    assert_eq!(tensor.numel(), 6);
    assert_eq!(tensor.ndim(), 2);
    assert_eq!(tensor.to_data(), data);
}

#[test]
fn test_different_data_types() {
    let shape = Shape::new([2, 2]);

    let float_tensor: Tensor<CpuBackend, 2, Float> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], shape.clone());

    let int_tensor: Tensor<CpuBackend, 2, Int> = Tensor::from_data(vec![1, 2, 3, 4], shape.clone());

    let bool_tensor: Tensor<CpuBackend, 2, Bool> =
        Tensor::from_data(vec![true, false, true, false], shape);

    assert_eq!(float_tensor.to_data(), vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(int_tensor.to_data(), vec![1, 2, 3, 4]);
    assert_eq!(bool_tensor.to_data(), vec![true, false, true, false]);
}

#[test]
fn test_factory_methods() {
    let shape = Shape::new([2, 3]);

    let zeros = Tensor::<CpuBackend, 2>::zeros(shape.clone());
    assert_eq!(zeros.to_data(), vec![0.0; 6]);

    let ones = Tensor::<CpuBackend, 2>::ones(shape.clone());
    assert_eq!(ones.to_data(), vec![1.0; 6]);

    let filled = Tensor::<CpuBackend, 2>::full(shape, 42.0);
    assert_eq!(filled.to_data(), vec![42.0; 6]);
}

#[test]
fn test_range_tensor() {
    let range = Tensor::<CpuBackend, 1>::range(0.0, 5.0, 1.0);
    assert_eq!(range.to_data(), vec![0.0, 1.0, 2.0, 3.0, 4.0]);

    let range_step = Tensor::<CpuBackend, 1>::range(0.0, 3.0, 0.5);
    assert_eq!(range_step.to_data(), vec![0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
}

#[test]
fn test_element_wise_operations() {
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2]));

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
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));

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
fn test_matrix_multiplication_square() {
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2]));

    let result = a.matmul(&b);

    // Manual calculation:
    // [1 2] * [5 6] = [1*5+2*7  1*6+2*8] = [19 22]
    // [3 4]   [7 8]   [3*5+4*7  3*6+4*8]   [43 50]
    assert_eq!(result.to_data(), vec![19.0, 22.0, 43.0, 50.0]);
    assert_eq!(result.shape().dims(), &[2, 2]);
}

#[test]
fn test_matrix_multiplication_rectangular() {
    let a: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3]));
    let b: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], Shape::new([3, 2]));

    let result = a.matmul(&b);

    // Manual calculation:
    // [1 2 3] * [7  8 ] = [1*7+2*9+3*11  1*8+2*10+3*12] = [58  64]
    // [4 5 6]   [9  10]   [4*7+5*9+6*11  4*8+5*10+6*12]   [139 154]
    //           [11 12]
    assert_eq!(result.to_data(), vec![58.0, 64.0, 139.0, 154.0]);
    assert_eq!(result.shape().dims(), &[2, 2]);
}

#[test]
fn test_mathematical_operations() {
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.0, 1.0, 2.0, 3.0], Shape::new([2, 2]));

    let sin_result = a.sin();
    let cos_result = a.cos();
    let exp_result = a.exp();
    let sqrt_result = a.sqrt();

    // Check sin(0) = 0, sin(1) ≈ 0.841
    assert!((sin_result.to_data()[0] - 0.0).abs() < 1e-6);
    assert!((sin_result.to_data()[1] - 1.0_f32.sin()).abs() < 1e-6);

    // Check cos(0) = 1
    assert!((cos_result.to_data()[0] - 1.0).abs() < 1e-6);

    // Check exp(0) = 1
    assert!((exp_result.to_data()[0] - 1.0).abs() < 1e-6);

    // Check sqrt values
    assert!((sqrt_result.to_data()[0] - 0.0).abs() < 1e-6);
    assert!((sqrt_result.to_data()[1] - 1.0).abs() < 1e-6);
}

#[test]
fn test_reduction_operations() {
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));

    assert_eq!(a.sum(), 10.0);
    assert_eq!(a.mean(), 2.5);
    assert_eq!(a.max(), 4.0);
    assert_eq!(a.min(), 1.0);
}

#[test]
fn test_multidimensional_tensors() {
    // 1D tensor
    let tensor_1d: Tensor<CpuBackend, 1> = Tensor::from_data(vec![1.0, 2.0, 3.0], Shape::new([3]));
    assert_eq!(tensor_1d.ndim(), 1);
    assert_eq!(tensor_1d.numel(), 3);

    // 3D tensor
    let tensor_3d: Tensor<CpuBackend, 3> = Tensor::from_data(
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        Shape::new([2, 2, 2]),
    );
    assert_eq!(tensor_3d.ndim(), 3);
    assert_eq!(tensor_3d.numel(), 8);
    assert_eq!(tensor_3d.shape().dim(0), 2);
    assert_eq!(tensor_3d.shape().dim(1), 2);
    assert_eq!(tensor_3d.shape().dim(2), 2);

    // 4D tensor
    let tensor_4d: Tensor<CpuBackend, 4> = Tensor::from_data(
        (1..=16).map(|x| x as f32).collect(),
        Shape::new([2, 2, 2, 2]),
    );
    assert_eq!(tensor_4d.ndim(), 4);
    assert_eq!(tensor_4d.numel(), 16);
}

#[test]
fn test_shape_properties() {
    let shape_2d = Shape::new([3, 4]);
    assert_eq!(shape_2d.dims(), &[3, 4]);
    assert_eq!(shape_2d.numel(), 12);
    assert_eq!(shape_2d.ndim(), 2);
    assert_eq!(shape_2d.dim(0), 3);
    assert_eq!(shape_2d.dim(1), 4);

    let shape_3d = Shape::new([2, 3, 4]);
    assert_eq!(shape_3d.numel(), 24);
    assert_eq!(shape_3d.ndim(), 3);
}

#[test]
fn test_tensor_cloning() {
    let original: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let cloned = original.clone();

    assert_eq!(original.to_data(), cloned.to_data());
    assert_eq!(original.shape().dims(), cloned.shape().dims());
}

#[test]
fn test_operation_chaining() {
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 1.0, 1.0, 1.0], Shape::new([2, 2]));

    // Chain operations: (a + b) * a
    let result = a.add(&b).mul(&a);

    // a + b = [2, 3, 4, 5]
    // (a + b) * a = [2*1, 3*2, 4*3, 5*4] = [2, 6, 12, 20]
    assert_eq!(result.to_data(), vec![2.0, 6.0, 12.0, 20.0]);
}

#[test]
fn test_complex_mathematical_chains() {
    let x: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));

    // Test complex chain: sin(x^2) + cos(x)
    let result = x.pow(2.0).sin().add(&x.cos());

    assert_eq!(result.numel(), 4);
    assert_eq!(result.shape().dims(), &[2, 2]);

    // Test mathematical properties
    let data = result.to_data();
    for val in data {
        assert!(val.is_finite()); // Should be finite values
    }
}

#[test]
fn test_large_tensor_operations() {
    let large_shape = Shape::new([100, 100]);
    let a = Tensor::<CpuBackend, 2>::ones(large_shape.clone());
    let b = Tensor::<CpuBackend, 2>::full(large_shape, 2.0);

    let sum = a.add(&b);
    assert_eq!(sum.numel(), 10000);
    assert!(sum.to_data().iter().all(|&x| (x - 3.0).abs() < 1e-6));

    let mean_val = sum.mean();
    assert!((mean_val - 3.0).abs() < 1e-6);
}

#[test]
fn test_tensor_debug_format() {
    let tensor: Tensor<CpuBackend, 2, Float> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));

    let debug_str = format!("{:?}", tensor);
    assert!(debug_str.contains("Tensor<Float, 2>"));
    assert!(debug_str.contains("[2, 2]"));
}

#[test]
fn test_shape_display() {
    let shape = Shape::new([2, 3, 4]);
    let display_str = format!("{}", shape);
    assert_eq!(display_str, "[2, 3, 4]");
}

#[test]
fn test_backend_operations() {
    use mini_burn::backend::{Backend, CpuBackend};

    let _backend = CpuBackend::new();
    let device = CpuBackend::default_device();

    let data = vec![1.0, 2.0, 3.0, 4.0];
    let storage = CpuBackend::from_data::<Float>(data.clone(), &device);

    assert_eq!(CpuBackend::to_data(&storage), data);
    assert_eq!(CpuBackend::storage_size(&storage), 4);
}

#[test]
fn test_advanced_matrix_operations() {
    // Test matrix multiplication with different sizes
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0], Shape::new([1, 3])); // 1x3
    let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![4.0, 5.0, 6.0], Shape::new([3, 1])); // 3x1

    let result = a.matmul(&b); // Should be 1x1
    assert_eq!(result.shape().dims(), &[1, 1]);
    assert_eq!(result.to_data(), vec![32.0]); // 1*4 + 2*5 + 3*6 = 32

    // Test with identity-like matrix
    let identity: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.0, 0.0, 0.0, 1.0], Shape::new([2, 2]));
    let test_matrix: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![3.0, 4.0, 5.0, 6.0], Shape::new([2, 2]));

    let result = test_matrix.matmul(&identity);
    assert_eq!(result.to_data(), vec![3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_power_and_root_operations() {
    let base: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.0, 4.0, 9.0, 16.0], Shape::new([2, 2]));

    let squared = base.pow(2.0);
    assert_eq!(squared.to_data(), vec![1.0, 16.0, 81.0, 256.0]);

    let sqrt_result = base.sqrt();
    assert_eq!(sqrt_result.to_data(), vec![1.0, 2.0, 3.0, 4.0]);

    let power_half = base.pow(0.5);
    let sqrt_data = sqrt_result.to_data();
    let power_data = power_half.to_data();
    for (sqrt_val, power_val) in sqrt_data.iter().zip(power_data.iter()) {
        assert!((sqrt_val - power_val).abs() < 1e-6);
    }
}

#[test]
fn test_absolute_value() {
    let mixed: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![-3.0, -1.0, 0.0, 2.0], Shape::new([2, 2]));
    let abs_result = mixed.abs();
    assert_eq!(abs_result.to_data(), vec![3.0, 1.0, 0.0, 2.0]);
}

#[test]
fn test_logarithmic_operations() {
    let positive: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1.0, 2.718281828, 7.389, 20.085], Shape::new([2, 2]));

    let ln_result = positive.ln();
    let exp_result = positive.exp();

    // ln(e) should be approximately 1
    assert!((ln_result.to_data()[1] - 1.0).abs() < 1e-5);

    // Check that exp(ln(x)) ≈ x for positive values
    let exp_ln = positive.ln().exp();
    let original_data = positive.to_data();
    let round_trip_data = exp_ln.to_data();

    for (orig, round_trip) in original_data.iter().zip(round_trip_data.iter()) {
        assert!((orig - round_trip).abs() < 1e-4);
    }
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
#[should_panic(expected = "Data length")]
fn test_invalid_data_length() {
    // This should panic: data length doesn't match shape
    let _: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0], Shape::new([2, 2]));
    // Need 4 elements, got 3
}

#[test]
fn test_comprehensive_workflow() {
    // Simulate a complete mini deep learning workflow

    // 1. Create input data (batch of 2D points)
    let input_data = vec![0.5, 0.3, 0.8, 0.1, 0.2, 0.9];
    let inputs: Tensor<CpuBackend, 2> = Tensor::from_data(input_data, Shape::new([3, 2])); // 3 samples, 2 features each

    // 2. Create weight matrix for simple linear transformation
    let weights: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![0.1, 0.2, 0.3, 0.4], Shape::new([2, 2]));
    let bias: Tensor<CpuBackend, 2> = Tensor::from_data(vec![0.1, 0.05], Shape::new([1, 2]));

    // 3. Forward pass: inputs @ weights + bias
    let linear_output = inputs.matmul(&weights);
    assert_eq!(linear_output.shape().dims(), &[3, 2]);

    // 4. Add bias (simplified - in real implementation would broadcast)
    let bias_expanded = bias.to_data();
    let output_data = linear_output.to_data();
    let biased_data: Vec<f32> = output_data
        .chunks(2)
        .flat_map(|chunk| vec![chunk[0] + bias_expanded[0], chunk[1] + bias_expanded[1]])
        .collect();

    let biased_output: Tensor<CpuBackend, 2> = Tensor::from_data(biased_data, Shape::new([3, 2]));

    // 5. Apply activation (sigmoid approximation using tensor ops)
    let activated = biased_output.mul_scalar(-1.0).exp().add_scalar(1.0);
    let sigmoid_like = Tensor::ones(Shape::new([3, 2])).div(&activated);

    // 6. Verify output properties
    assert_eq!(sigmoid_like.numel(), 6);
    assert!(sigmoid_like.to_data().iter().all(|&x| x >= 0.0 && x <= 1.0));

    // 7. Compute some statistics
    let mean_activation = sigmoid_like.mean();
    let max_activation = sigmoid_like.max();
    let min_activation = sigmoid_like.min();

    assert!(mean_activation >= 0.0 && mean_activation <= 1.0);
    assert!(max_activation >= mean_activation);
    assert!(min_activation <= mean_activation);
}

#[test]
fn test_tensor_equality_semantics() {
    let a: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let b: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    let c: Tensor<CpuBackend, 2> = Tensor::from_data(vec![1.0, 2.0, 3.0, 5.0], Shape::new([2, 2]));

    // Test data equality
    assert_eq!(a.to_data(), b.to_data());
    assert_ne!(a.to_data(), c.to_data());

    // Test shape equality
    assert_eq!(a.shape(), b.shape());
    assert_eq!(a.shape(), c.shape());
}

#[test]
fn test_numerical_stability() {
    // Test operations near numerical limits
    let small_values: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1e-7, 1e-6, 1e-5, 1e-4], Shape::new([2, 2]));
    let large_values: Tensor<CpuBackend, 2> =
        Tensor::from_data(vec![1e4, 1e5, 1e6, 1e7], Shape::new([2, 2]));

    // These operations should not panic or produce NaN/Inf
    let sqrt_small = small_values.sqrt();
    let ln_small = small_values.ln();
    let exp_small = small_values.exp();

    // Verify results are finite
    assert!(sqrt_small.to_data().iter().all(|x| x.is_finite()));
    assert!(ln_small.to_data().iter().all(|x| x.is_finite()));
    assert!(exp_small.to_data().iter().all(|x| x.is_finite()));

    // Test with large values
    let small_div_large = small_values.div(&large_values);
    assert!(small_div_large.to_data().iter().all(|x| x.is_finite()));
    assert!(small_div_large.to_data().iter().all(|&x| x >= 0.0));
}
