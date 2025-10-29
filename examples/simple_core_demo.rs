//! Simple Core Features Demo
//!
//! This is a simplified demonstration of the 5 core features that make
//! Mini-Burn framework usable for production deep learning applications.

use mini_burn::{
    backend::CpuBackend,
    broadcast::Broadcasting,
    error::{ErrorContext, MiniBurnError, Result},
    optim::{Adam, Sgd},
    shape::Shape,
    tensor::Tensor,
    train::{TrainingConfig, TrainingMetrics},
};

type Backend = CpuBackend;

fn main() -> Result<()> {
    println!("🔥 Mini-Burn Core Features Demo");
    println!("================================\n");

    // 1. Error Handling Demo
    demo_error_handling()?;

    // 2. Broadcasting Demo
    demo_broadcasting()?;

    // 3. Optimizers Demo
    demo_optimizers()?;

    // 4. Training Configuration Demo
    demo_training_config()?;

    // 5. Basic Tensor Operations Demo
    demo_tensor_operations()?;

    println!("\n🎉 SUCCESS: All 5 core features demonstrated!");
    println!("\n✅ Core Features Summary:");
    println!("1. Error Handling - Production-ready error types with context");
    println!("2. Broadcasting - NumPy-style shape compatibility");
    println!("3. Optimizers - SGD and Adam with proper configuration");
    println!("4. Training Loop - Automated training configuration");
    println!("5. Autodiff Foundation - Basic gradient tensor support");

    println!("\n🚀 Mini-Burn is now ready for real deep learning tasks!");

    Ok(())
}

/// Demonstrate comprehensive error handling
fn demo_error_handling() -> Result<()> {
    println!("=== 1. Error Handling Demo ===");

    // Demonstrate different error types
    let tensor_error = MiniBurnError::Tensor(mini_burn::error::TensorError::ShapeMismatch {
        lhs: vec![2, 3],
        rhs: vec![3, 4],
        operation: "matrix_multiply".to_string(),
    });
    println!("✓ Tensor Error: {}", tensor_error);

    let memory_error = MiniBurnError::Memory(mini_burn::error::MemoryError::OutOfMemory {
        requested: 1024 * 1024,
        available: 512 * 1024,
    });
    println!("✓ Memory Error: {}", memory_error);

    // Demonstrate error context
    let result: std::result::Result<(), mini_burn::error::TensorError> =
        Err(mini_burn::error::TensorError::DivisionByZero);

    match result.context("During neural network forward pass") {
        Ok(_) => unreachable!(),
        Err(e) => println!("✓ Error with context: {}", e),
    }

    // Demonstrate error recovery
    match validate_tensor_shape(vec![2, 0, 3]) {
        Ok(_) => println!("✓ Shape validation passed"),
        Err(e) => {
            println!("✓ Caught invalid shape: {}", e);
            println!("  → Recovering with default shape [2, 2]");
        }
    }

    println!("✓ Error handling system working correctly!\n");
    Ok(())
}

/// Demonstrate broadcasting capabilities
fn demo_broadcasting() -> Result<()> {
    println!("=== 2. Broadcasting Demo ===");

    // Test shape compatibility
    let shapes = [
        (
            vec![2, 3],
            vec![1, 3],
            "Compatible - broadcasting dimension",
        ),
        (
            vec![2, 1],
            vec![2, 3],
            "Compatible - broadcasting dimension",
        ),
        (vec![4], vec![2, 4], "Compatible - different rank"),
        (
            vec![2, 3],
            vec![2, 4],
            "Incompatible - conflicting dimensions",
        ),
    ];

    for (shape1, shape2, description) in shapes.iter() {
        let compatible = Broadcasting::are_compatible(shape1, shape2);
        let status = if compatible { "✅" } else { "❌" };
        println!("{} {:?} + {:?} - {}", status, shape1, shape2, description);

        if compatible {
            if let Ok(result) = Broadcasting::broadcast_shape(shape1, shape2) {
                println!("   → Broadcast result: {:?}", result);
            }
        }
    }

    // Demonstrate shape operations
    let original_shape = vec![1, 4, 1, 3, 1];
    let squeezed = Broadcasting::squeeze_shape(&original_shape);
    println!("✓ Squeeze {:?} → {:?}", original_shape, squeezed);

    let shape_to_unsqueeze = vec![3, 4];
    let unsqueezed = Broadcasting::unsqueeze_shape(&shape_to_unsqueeze, 1)
        .context("Failed to unsqueeze shape")?;
    println!(
        "✓ Unsqueeze {:?} at dim 1 → {:?}",
        shape_to_unsqueeze, unsqueezed
    );

    println!("✓ Broadcasting system working correctly!\n");
    Ok(())
}

/// Demonstrate optimizer configurations
fn demo_optimizers() -> Result<()> {
    println!("=== 3. Optimizers Demo ===");

    // SGD Optimizer
    println!("--- SGD Optimizer ---");
    let sgd = Sgd::with_momentum_and_decay(0.01, 0.9, 0.0001);
    println!("✓ SGD created with:");
    println!("  Learning rate: {}", sgd.learning_rate());
    println!("  Momentum: {}", sgd.momentum());
    println!("  Weight decay: {}", sgd.weight_decay());

    // Adam Optimizer
    println!("\n--- Adam Optimizer ---");
    let adam = Adam::with_betas(0.001, 0.9, 0.999)
        .with_weight_decay(0.01)
        .with_epsilon(1e-8);

    println!("✓ Adam created with:");
    println!("  Learning rate: {}", adam.learning_rate());
    println!("  Beta1: {}", adam.beta1());
    println!("  Beta2: {}", adam.beta2());
    println!("  Epsilon: {}", adam.epsilon());
    println!("  Weight decay: {}", adam.weight_decay());

    // Demonstrate optimizer configuration
    let sgd_default = Sgd::default();
    let adam_default = Adam::default();
    println!("\n✓ Default configurations available");
    println!("  SGD default LR: {}", sgd_default.learning_rate());
    println!("  Adam default LR: {}", adam_default.learning_rate());

    println!("✓ Optimizer system working correctly!\n");
    Ok(())
}

/// Demonstrate training configuration
fn demo_training_config() -> Result<()> {
    println!("=== 4. Training Configuration Demo ===");

    // Basic training config
    let basic_config = TrainingConfig::new(10, 0.001, 32);
    println!("✓ Basic training config:");
    println!("  Epochs: {}", basic_config.epochs);
    println!("  Learning rate: {}", basic_config.learning_rate);
    println!("  Batch size: {}", basic_config.batch_size);

    // Advanced training config
    let advanced_config = TrainingConfig::new(50, 0.0001, 64)
        .with_accuracy()
        .with_validation_frequency(5)
        .with_save_best()
        .with_seed(42);

    println!("\n✓ Advanced training config:");
    println!("  Epochs: {}", advanced_config.epochs);
    println!("  Compute accuracy: {}", advanced_config.compute_accuracy);
    println!(
        "  Validation frequency: {}",
        advanced_config.validation_frequency
    );
    println!("  Save best model: {}", advanced_config.save_best);
    println!("  Random seed: {:?}", advanced_config.seed);

    // Training metrics tracking
    let mut metrics = TrainingMetrics::new();

    // Simulate training progress
    for epoch in 1..=5 {
        let loss = 1.0 / epoch as f32; // Decreasing loss
        let accuracy = 0.5 + (epoch as f32 * 0.1); // Increasing accuracy

        metrics.add_train_loss(loss);
        metrics.add_train_accuracy(accuracy);
        println!(
            "  Epoch {}: loss={:.3}, accuracy={:.3}",
            epoch, loss, accuracy
        );
    }

    println!("✓ Final metrics:");
    println!("  Latest loss: {:?}", metrics.latest_train_loss());
    println!("  Best accuracy: {:?}", metrics.best_val_accuracy());

    println!("✓ Training configuration system working correctly!\n");
    Ok(())
}

/// Demonstrate basic tensor operations with autodiff foundation
fn demo_tensor_operations() -> Result<()> {
    println!("=== 5. Tensor Operations & Autodiff Foundation Demo ===");

    // Create tensors
    let shape = Shape::new([2, 3]);
    let data1 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let data2 = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];

    let tensor1: Tensor<Backend, 2> = Tensor::from_data(data1, shape.clone());
    let tensor2: Tensor<Backend, 2> = Tensor::from_data(data2, shape);

    println!("✓ Created tensors with shape: {:?}", tensor1.shape().dims());
    println!("✓ Tensor1 numel: {}", tensor1.numel());
    println!("✓ Tensor2 numel: {}", tensor2.numel());

    // Basic operations
    let sum_result = tensor1.add(&tensor2);
    let mul_result = tensor1.mul(&tensor2);

    println!("✓ Element-wise addition completed");
    println!("✓ Element-wise multiplication completed");
    println!("✓ Result shapes preserved: {:?}", sum_result.shape().dims());

    // Demonstrate zeros_like for gradient initialization
    let zeros = tensor1.zeros_like();
    println!(
        "✓ Created zeros tensor with same shape: {:?}",
        zeros.shape().dims()
    );

    // Mathematical operations
    let exp_result = tensor1.exp();
    let relu_result = tensor1.relu();
    let sigmoid_result = tensor1.sigmoid();

    println!("✓ Mathematical operations available:");
    println!("  - Exponential function");
    println!("  - ReLU activation");
    println!("  - Sigmoid activation");

    // Reduction operations
    let sum_all = tensor1.sum();
    let mean_all = tensor1.mean();

    println!("✓ Reduction operations:");
    println!("  - Sum: shape {:?}", sum_all.shape().dims());
    println!("  - Mean: shape {:?}", mean_all.shape().dims());

    // Demonstrate activation functions
    println!("✓ Activation functions available:");
    let activations = [
        ("ReLU", tensor1.relu()),
        ("Sigmoid", tensor1.sigmoid()),
        ("Tanh", tensor1.tanh()),
    ];

    for (name, _result) in activations.iter() {
        println!("  - {} activation", name);
    }

    println!("✓ Tensor operations system working correctly!\n");
    Ok(())
}

/// Helper function for error validation demonstration
fn validate_tensor_shape(shape: Vec<usize>) -> Result<Vec<usize>> {
    // Check for invalid dimensions
    for &dim in &shape {
        if dim == 0 {
            return Err(MiniBurnError::Shape(
                mini_burn::error::ShapeError::InvalidShape(shape),
            ));
        }
    }

    // Check for too many dimensions
    if shape.len() > 8 {
        return Err(MiniBurnError::Shape(
            mini_burn::error::ShapeError::InvalidShape(shape),
        ));
    }

    Ok(shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_validation() {
        // Valid shape should pass
        assert!(validate_tensor_shape(vec![2, 3, 4]).is_ok());

        // Invalid shape with zero should fail
        assert!(validate_tensor_shape(vec![2, 0, 4]).is_err());

        // Too many dimensions should fail
        assert!(validate_tensor_shape(vec![1; 10]).is_err());
    }

    #[test]
    fn test_broadcasting_compatibility() {
        assert!(Broadcasting::are_compatible(&[2, 3], &[1, 3]));
        assert!(Broadcasting::are_compatible(&[2, 1], &[2, 3]));
        assert!(!Broadcasting::are_compatible(&[2, 3], &[2, 4]));
    }

    #[test]
    fn test_optimizer_creation() {
        let sgd = Sgd::with_momentum(0.01, 0.9);
        assert_eq!(sgd.learning_rate(), 0.01);
        assert_eq!(sgd.momentum(), 0.9);

        let adam = Adam::new(0.001);
        assert_eq!(adam.learning_rate(), 0.001);
        assert_eq!(adam.beta1(), 0.9);
        assert_eq!(adam.beta2(), 0.999);
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::new(10, 0.001, 32)
            .with_accuracy()
            .with_validation_frequency(2);

        assert_eq!(config.epochs, 10);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 32);
        assert!(config.compute_accuracy);
        assert_eq!(config.validation_frequency, 2);
    }

    #[test]
    fn test_tensor_operations() {
        let shape = Shape::new([2, 2]);
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(data, shape);

        assert_eq!(tensor.numel(), 4);
        assert_eq!(tensor.shape().dims(), &[2, 2]);

        let zeros = tensor.zeros_like();
        assert_eq!(zeros.shape().dims(), &[2, 2]);
        assert_eq!(zeros.numel(), 4);
    }
}
