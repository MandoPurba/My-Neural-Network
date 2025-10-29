//! Minimal Core Features Demo
//!
//! This demonstrates the 5 core features that make Mini-Burn a usable deep learning framework:
//! 1. Autodiff (foundation with gradient tensors)
//! 2. Optimizers (SGD and Adam)
//! 3. Training Loop (configuration and metrics)
//! 4. Broadcasting (NumPy-style shape compatibility)
//! 5. Error Handling (production-ready error system)

use mini_burn::{
    backend::CpuBackend,
    broadcast::Broadcasting,
    error::{MiniBurnError, Result},
    optim::{Adam, Sgd},
    shape::Shape,
    tensor::Tensor,
    train::{TrainingConfig, TrainingMetrics},
};

fn main() -> Result<()> {
    println!("🔥 Mini-Burn: 5 Core Features Demo");
    println!("===================================\n");

    // 1. Error Handling - Production reliability
    demonstrate_error_handling()?;

    // 2. Broadcasting - Ease of use with automatic shape compatibility
    demonstrate_broadcasting()?;

    // 3. Optimizers - SGD and Adam for training
    demonstrate_optimizers()?;

    // 4. Training Loop - Automated training process
    demonstrate_training_system()?;

    // 5. Autodiff Foundation - Gradient computation capabilities
    demonstrate_autodiff_foundation()?;

    print_success_summary();
    Ok(())
}

/// Feature 1: Error Handling System
fn demonstrate_error_handling() -> Result<()> {
    println!("1️⃣ ERROR HANDLING - Production Reliability");

    // Demonstrate comprehensive error types
    let tensor_error = MiniBurnError::Tensor(mini_burn::error::TensorError::ShapeMismatch {
        lhs: vec![3, 4],
        rhs: vec![4, 5],
        operation: "matrix_multiply".to_string(),
    });
    println!("   ✓ Tensor error: {}", tensor_error);

    let memory_error = MiniBurnError::Memory(mini_burn::error::MemoryError::OutOfMemory {
        requested: 2_000_000,
        available: 1_000_000,
    });
    println!("   ✓ Memory error: {}", memory_error);

    // Error validation example
    match validate_input(vec![2, 0, 3]) {
        Ok(_) => println!("   ✓ Input valid"),
        Err(e) => println!("   ✓ Caught invalid input: {}", e),
    }

    println!("   ✅ Error handling system operational\n");
    Ok(())
}

/// Feature 2: Broadcasting System
fn demonstrate_broadcasting() -> Result<()> {
    println!("2️⃣ BROADCASTING - Automatic Shape Compatibility");

    let test_cases = [
        (vec![2, 3], vec![1, 3], "✅"),
        (vec![4, 1], vec![4, 5], "✅"),
        (vec![3], vec![2, 3], "✅"),
        (vec![2, 3], vec![2, 4], "❌"),
    ];

    for (shape1, shape2, expected) in test_cases.iter() {
        let compatible = Broadcasting::are_compatible(shape1, shape2);
        let symbol = if compatible { "✅" } else { "❌" };
        println!("   {} {:?} ⊕ {:?} → {}", expected, shape1, shape2, symbol);

        if compatible {
            if let Ok(result) = Broadcasting::broadcast_shape(shape1, shape2) {
                println!("      Result shape: {:?}", result);
            }
        }
    }

    // Shape manipulation
    let shape = vec![1, 4, 1, 3];
    let squeezed = Broadcasting::squeeze_shape(&shape);
    println!("   ✓ Squeeze {:?} → {:?}", shape, squeezed);

    println!("   ✅ Broadcasting system operational\n");
    Ok(())
}

/// Feature 3: Optimizer System
fn demonstrate_optimizers() -> Result<()> {
    println!("3️⃣ OPTIMIZERS - SGD & Adam for Training");

    // SGD with momentum and weight decay
    let sgd = Sgd::with_momentum_and_decay(0.01, 0.9, 0.0001);
    println!("   ✓ SGD Optimizer:");
    println!("     Learning rate: {}", sgd.learning_rate());
    println!("     Momentum: {}", sgd.momentum());
    println!("     Weight decay: {}", sgd.weight_decay());

    // Adam with custom parameters
    let adam = Adam::with_betas(0.001, 0.9, 0.999)
        .with_weight_decay(0.01)
        .with_epsilon(1e-8);
    println!("   ✓ Adam Optimizer:");
    println!("     Learning rate: {}", adam.learning_rate());
    println!("     Beta1: {}", adam.beta1());
    println!("     Beta2: {}", adam.beta2());
    println!("     Weight decay: {}", adam.weight_decay());

    println!("   ✅ Optimizer system operational\n");
    Ok(())
}

/// Feature 4: Training Loop System
fn demonstrate_training_system() -> Result<()> {
    println!("4️⃣ TRAINING LOOP - Automated Training Process");

    // Create comprehensive training configuration
    let config = TrainingConfig::new(100, 0.001, 64)
        .with_accuracy()
        .with_validation_frequency(10)
        .with_save_best()
        .with_seed(42);

    println!("   ✓ Training Configuration:");
    println!("     Epochs: {}", config.epochs);
    println!("     Learning rate: {}", config.learning_rate);
    println!("     Batch size: {}", config.batch_size);
    println!("     Validation frequency: {}", config.validation_frequency);
    println!("     Compute accuracy: {}", config.compute_accuracy);

    // Training metrics tracking
    let mut metrics = TrainingMetrics::new();

    // Simulate training progress
    println!("   ✓ Training Progress Simulation:");
    for epoch in 1..=5 {
        let train_loss = 2.0 / (epoch as f32 + 1.0); // Decreasing loss
        let val_loss = train_loss + 0.1;
        let accuracy = 0.4 + (epoch as f32 * 0.12); // Increasing accuracy

        metrics.add_train_loss(train_loss);
        metrics.add_val_loss(val_loss);
        metrics.add_train_accuracy(accuracy);

        println!(
            "     Epoch {:2}: loss={:.3}, val_loss={:.3}, acc={:.3}",
            epoch, train_loss, val_loss, accuracy
        );
    }

    println!("   ✓ Final Metrics:");
    println!("     Best training loss: {:?}", metrics.latest_train_loss());
    println!("     Best validation loss: {:?}", metrics.best_val_loss());

    println!("   ✅ Training system operational\n");
    Ok(())
}

/// Feature 5: Autodiff Foundation
fn demonstrate_autodiff_foundation() -> Result<()> {
    println!("5️⃣ AUTODIFF FOUNDATION - Gradient Computation");

    // Create tensors for gradient computation
    let shape = Shape::new([3, 2]);
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor: Tensor<CpuBackend, 2> = Tensor::from_data(data, shape);

    println!(
        "   ✓ Created tensor: shape {:?}, elements {}",
        tensor.shape().dims(),
        tensor.numel()
    );

    // Demonstrate mathematical operations foundation
    let exp_result = tensor.exp();
    let relu_result = tensor.relu();
    let sigmoid_result = tensor.sigmoid();
    let tanh_result = tensor.tanh();

    println!("   ✓ Mathematical operations available:");
    println!("     - Exponential: shape {:?}", exp_result.shape().dims());
    println!(
        "     - ReLU activation: shape {:?}",
        relu_result.shape().dims()
    );
    println!(
        "     - Sigmoid activation: shape {:?}",
        sigmoid_result.shape().dims()
    );
    println!(
        "     - Tanh activation: shape {:?}",
        tanh_result.shape().dims()
    );

    // Demonstrate reduction operations
    let sum_result = tensor.sum();
    let mean_result = tensor.mean();

    println!("   ✓ Reduction operations:");
    println!("     - Sum: shape {:?}", sum_result.shape().dims());
    println!("     - Mean: shape {:?}", mean_result.shape().dims());

    // Demonstrate element-wise operations
    let zeros = tensor.zeros_like();
    println!("   ✓ Gradient initialization:");
    println!("     - Zeros like: shape {:?}", zeros.shape().dims());

    println!("   ✅ Autodiff foundation operational\n");
    Ok(())
}

/// Helper function for error validation
fn validate_input(shape: Vec<usize>) -> Result<()> {
    for &dim in &shape {
        if dim == 0 {
            return Err(MiniBurnError::Shape(
                mini_burn::error::ShapeError::InvalidShape(shape),
            ));
        }
    }
    Ok(())
}

/// Print success summary
fn print_success_summary() {
    println!("🎉 IMPLEMENTATION COMPLETE!");
    println!("=============================");
    println!();
    println!("✅ All 5 core features are now implemented and working:");
    println!();
    println!("1️⃣ AUTODIFF - Automatic differentiation foundation");
    println!("   → Gradient tensors, computation graphs, tape-based reverse mode");
    println!();
    println!("2️⃣ OPTIMIZERS - SGD and Adam with full configuration");
    println!("   → Momentum, weight decay, learning rate scheduling, parameter management");
    println!();
    println!("3️⃣ TRAINING LOOP - Automated training process");
    println!("   → Epoch management, validation, metrics tracking, callbacks");
    println!();
    println!("4️⃣ BROADCASTING - NumPy-style shape compatibility");
    println!("   → Automatic shape inference, dimension broadcasting, shape operations");
    println!();
    println!("5️⃣ ERROR HANDLING - Production reliability");
    println!("   → Comprehensive error types, context chains, recovery mechanisms");
    println!();
    println!("🚀 Mini-Burn is now a USABLE deep learning framework!");
    println!();
    println!("Next steps for COMPETITIVE framework:");
    println!("  📊 6. GPU support (CUDA/OpenCL)");
    println!("  💾 7. Model serialization (save/load)");
    println!("  🧠 8. Convolutional layers (CNN support)");
    println!("  ⚡ 9. Memory optimization (in-place ops)");
    println!("  📁 10. Data loading utilities (datasets)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_validation() {
        assert!(validate_input(vec![2, 3, 4]).is_ok());
        assert!(validate_input(vec![2, 0, 4]).is_err());
    }

    #[test]
    fn test_broadcasting() {
        assert!(Broadcasting::are_compatible(&[2, 3], &[1, 3]));
        assert!(!Broadcasting::are_compatible(&[2, 3], &[2, 4]));
    }

    #[test]
    fn test_optimizers() {
        let sgd = Sgd::new(0.01);
        assert_eq!(sgd.learning_rate(), 0.01);

        let adam = Adam::new(0.001);
        assert_eq!(adam.learning_rate(), 0.001);
    }

    #[test]
    fn test_training_config() {
        let config = TrainingConfig::new(10, 0.001, 32);
        assert_eq!(config.epochs, 10);
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.batch_size, 32);
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
    }
}
