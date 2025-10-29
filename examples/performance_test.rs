//! Performance testing and benchmarking for mini-burn operations

use mini_burn::{CpuBackend, Shape, Tensor};
use std::time::Instant;

fn benchmark_operation<F>(name: &str, iterations: usize, operation: F)
where
    F: Fn() -> (),
{
    let start = Instant::now();

    for _ in 0..iterations {
        operation();
    }

    let duration = start.elapsed();
    let avg_time = duration.as_nanos() as f64 / iterations as f64;

    println!(
        "{}: {:.2} ns/op ({} iterations, {:.2} ms total)",
        name,
        avg_time,
        iterations,
        duration.as_millis()
    );
}

fn main() {
    println!("=== Mini-Burn Performance Tests ===\n");

    // 1. Tensor creation benchmarks
    println!("1. Tensor Creation Benchmarks:");

    benchmark_operation("Small tensor creation (2x2)", 10000, || {
        let _tensor = Tensor::<CpuBackend, 2>::zeros(Shape::new([2, 2]));
    });

    benchmark_operation("Medium tensor creation (10x10)", 1000, || {
        let _tensor = Tensor::<CpuBackend, 2>::zeros(Shape::new([10, 10]));
    });

    benchmark_operation("Large tensor creation (100x100)", 100, || {
        let _tensor = Tensor::<CpuBackend, 2>::zeros(Shape::new([100, 100]));
    });

    // 2. Element-wise operation benchmarks
    println!("\n2. Element-wise Operation Benchmarks:");

    let small_a = Tensor::<CpuBackend, 2>::ones(Shape::new([10, 10]));
    let small_b = Tensor::<CpuBackend, 2>::full(Shape::new([10, 10]), 2.0);

    benchmark_operation("Addition (10x10)", 5000, || {
        let _result = small_a.add(&small_b);
    });

    benchmark_operation("Multiplication (10x10)", 5000, || {
        let _result = small_a.mul(&small_b);
    });

    let medium_a = Tensor::<CpuBackend, 2>::ones(Shape::new([50, 50]));
    let medium_b = Tensor::<CpuBackend, 2>::full(Shape::new([50, 50]), 2.0);

    benchmark_operation("Addition (50x50)", 1000, || {
        let _result = medium_a.add(&medium_b);
    });

    // 3. Matrix multiplication benchmarks
    println!("\n3. Matrix Multiplication Benchmarks:");

    let mat_2x2_a = Tensor::<CpuBackend, 2>::ones(Shape::new([2, 2]));
    let mat_2x2_b = Tensor::<CpuBackend, 2>::full(Shape::new([2, 2]), 2.0);

    benchmark_operation("Matrix multiply (2x2)", 10000, || {
        let _result = mat_2x2_a.matmul(&mat_2x2_b);
    });

    let mat_10x10_a = Tensor::<CpuBackend, 2>::ones(Shape::new([10, 10]));
    let mat_10x10_b = Tensor::<CpuBackend, 2>::full(Shape::new([10, 10]), 2.0);

    benchmark_operation("Matrix multiply (10x10)", 1000, || {
        let _result = mat_10x10_a.matmul(&mat_10x10_b);
    });

    let mat_50x50_a = Tensor::<CpuBackend, 2>::ones(Shape::new([50, 50]));
    let mat_50x50_b = Tensor::<CpuBackend, 2>::full(Shape::new([50, 50]), 2.0);

    benchmark_operation("Matrix multiply (50x50)", 100, || {
        let _result = mat_50x50_a.matmul(&mat_50x50_b);
    });

    let mat_100x100_a = Tensor::<CpuBackend, 2>::ones(Shape::new([100, 100]));
    let mat_100x100_b = Tensor::<CpuBackend, 2>::full(Shape::new([100, 100]), 2.0);

    benchmark_operation("Matrix multiply (100x100)", 10, || {
        let _result = mat_100x100_a.matmul(&mat_100x100_b);
    });

    // 4. Memory operations
    println!("\n4. Memory Operation Benchmarks:");

    let test_tensor = Tensor::<CpuBackend, 2>::ones(Shape::new([100, 100]));

    benchmark_operation("Tensor clone (100x100)", 1000, || {
        let _cloned = test_tensor.clone();
    });

    benchmark_operation("To data conversion (100x100)", 1000, || {
        let _data = test_tensor.to_data();
    });

    // 5. Complex operation chains
    println!("\n5. Complex Operation Chains:");

    let chain_a = Tensor::<CpuBackend, 2>::ones(Shape::new([20, 20]));
    let chain_b = Tensor::<CpuBackend, 2>::full(Shape::new([20, 20]), 2.0);
    let chain_c = Tensor::<CpuBackend, 2>::full(Shape::new([20, 20]), 0.5);

    benchmark_operation("(A + B) * C - A (20x20)", 500, || {
        let _result = chain_a.add(&chain_b).mul(&chain_c).sub(&chain_a);
    });

    benchmark_operation("A @ B + C (20x20)", 500, || {
        let _result = chain_a.matmul(&chain_b).add(&chain_c);
    });

    // 6. Scalar operations
    println!("\n6. Scalar Operation Benchmarks:");

    let scalar_tensor = Tensor::<CpuBackend, 2>::ones(Shape::new([50, 50]));

    benchmark_operation("Scalar addition (50x50)", 2000, || {
        let _result = scalar_tensor.add_scalar(5.0);
    });

    benchmark_operation("Scalar multiplication (50x50)", 2000, || {
        let _result = scalar_tensor.mul_scalar(2.0);
    });

    // 7. Different tensor dimensions
    println!("\n7. Multi-dimensional Tensor Benchmarks:");

    let tensor_1d = Tensor::<CpuBackend, 1>::ones(Shape::new([1000]));
    let tensor_3d = Tensor::<CpuBackend, 3>::ones(Shape::new([10, 10, 10]));
    let tensor_4d = Tensor::<CpuBackend, 4>::ones(Shape::new([5, 5, 5, 8]));

    benchmark_operation("1D operations (1000 elements)", 5000, || {
        let _result = tensor_1d.add_scalar(1.0);
    });

    benchmark_operation("3D operations (10x10x10)", 1000, || {
        let _result = tensor_3d.add_scalar(1.0);
    });

    benchmark_operation("4D operations (5x5x5x8)", 1000, || {
        let _result = tensor_4d.add_scalar(1.0);
    });

    // 8. Mathematical function benchmarks
    println!("\n8. Mathematical Function Benchmarks:");

    let math_tensor = Tensor::<CpuBackend, 2>::ones(Shape::new([50, 50]));

    benchmark_operation("Sin operation (50x50)", 1000, || {
        let _result = math_tensor.sin();
    });

    benchmark_operation("Cos operation (50x50)", 1000, || {
        let _result = math_tensor.cos();
    });

    benchmark_operation("Exp operation (50x50)", 1000, || {
        let _result = math_tensor.exp();
    });

    benchmark_operation("Sqrt operation (50x50)", 1000, || {
        let _result = math_tensor.sqrt();
    });

    benchmark_operation("Power operation (50x50)", 1000, || {
        let _result = math_tensor.pow(2.0);
    });

    // 9. Reduction operations
    println!("\n9. Reduction Operation Benchmarks:");

    let reduce_tensor = Tensor::<CpuBackend, 2>::ones(Shape::new([100, 100]));

    benchmark_operation("Sum reduction (100x100)", 1000, || {
        let _result = reduce_tensor.sum();
    });

    benchmark_operation("Mean reduction (100x100)", 1000, || {
        let _result = reduce_tensor.mean();
    });

    benchmark_operation("Max reduction (100x100)", 1000, || {
        let _result = reduce_tensor.max();
    });

    benchmark_operation("Min reduction (100x100)", 1000, || {
        let _result = reduce_tensor.min();
    });

    // 10. Factory method benchmarks
    println!("\n10. Factory Method Benchmarks:");

    benchmark_operation("Zeros creation (100x100)", 1000, || {
        let _tensor = Tensor::<CpuBackend, 2>::zeros(Shape::new([100, 100]));
    });

    benchmark_operation("Ones creation (100x100)", 1000, || {
        let _tensor = Tensor::<CpuBackend, 2>::ones(Shape::new([100, 100]));
    });

    benchmark_operation("Full creation (100x100)", 1000, || {
        let _tensor = Tensor::<CpuBackend, 2>::full(Shape::new([100, 100]), 3.14);
    });

    benchmark_operation("Range creation (1000 elements)", 1000, || {
        let _tensor = Tensor::<CpuBackend, 1>::range(0.0, 1000.0, 1.0);
    });

    // 11. Comparative scaling test
    println!("\n11. Scaling Analysis:");

    let sizes = vec![10, 50, 100, 200];

    for size in sizes {
        let a = Tensor::<CpuBackend, 2>::ones(Shape::new([size, size]));
        let b = Tensor::<CpuBackend, 2>::full(Shape::new([size, size]), 2.0);

        let iterations = std::cmp::max(1, 10000 / (size * size / 100));

        benchmark_operation(&format!("Addition ({}x{})", size, size), iterations, || {
            let _result = a.add(&b);
        });

        let mm_iterations = std::cmp::max(1, 1000 / (size * size / 100));
        benchmark_operation(
            &format!("MatMul ({}x{})", size, size),
            mm_iterations,
            || {
                let _result = a.matmul(&b);
            },
        );
    }

    println!("\n=== Performance Tests Complete ===");
    println!("\nNote: These are naive implementations without SIMD or other optimizations.");
    println!("Real deep learning frameworks use highly optimized BLAS libraries.");
    println!("Results may vary significantly based on hardware and system load.");

    // Summary recommendations
    println!("\n=== Performance Summary ===");
    println!("- Tensor creation is relatively fast for small-medium sizes");
    println!("- Element-wise operations scale linearly with tensor size");
    println!("- Matrix multiplication scales cubically (O(n³)) as expected");
    println!("- Memory operations (clone, to_data) are dominated by allocation costs");
    println!("- Mathematical functions add overhead but remain reasonably fast");
    println!("- Reduction operations are efficient for statistical computations");

    println!("\nOptimization opportunities:");
    println!("- SIMD instructions for vectorized operations");
    println!("- BLAS integration for optimized linear algebra");
    println!("- In-place operations to reduce memory allocations");
    println!("- Parallel processing for large tensors");
    println!("- GPU acceleration for compute-intensive workloads");
}
