//! Core Features Demonstration
//!
//! This example demonstrates all 5 core features that make Mini-Burn framework usable:
//! 1. Autodiff (Automatic Differentiation)
//! 2. Optimizers (SGD and Adam)
//! 3. Training Loop (Automated training process)
//! 4. Broadcasting (for ease of use)
//! 5. Error Handling (production reliability)

use mini_burn::{
    autodiff::{no_grad, with_grad, GradTensor},
    backend::CpuBackend,
    broadcast::Broadcasting,
    error::{ErrorContext, MiniBurnError, Result},
    nn::{
        layers::{Linear, ReLU},
        loss::MSELoss,
        Module,
    },
    optim::{Adam, Optimizer, Sgd, TensorParameter},
    shape::Shape,
    tensor::Tensor,
    train::{
        EpochResult, LoggingCallback, TrainingCallback, TrainingConfig, TrainingLoop,
        TrainingMetrics,
    },
};

type Backend = CpuBackend;

/// Simple neural network for demonstration
struct SimpleNet {
    linear1: Linear<Backend>,
    relu: ReLU<Backend>,
    linear2: Linear<Backend>,
    training_mode: bool,
}

impl SimpleNet {
    fn new() -> Result<Self> {
        let linear1 = Linear::new(2, 4).context("Failed to create first linear layer")?;
        let relu = ReLU::new();
        let linear2 = Linear::new(4, 1).context("Failed to create second linear layer")?;

        Ok(Self {
            linear1,
            relu,
            linear2,
            training_mode: true,
        })
    }

    fn forward(&mut self, input: &Tensor<Backend, 2>) -> Result<Tensor<Backend, 2>> {
        // Demonstrate error handling with proper context
        let x = self
            .linear1
            .forward(input)
            .context("Forward pass failed at linear1")?;

        let x = self
            .relu
            .forward(&x)
            .context("Forward pass failed at relu")?;

        let output = self
            .linear2
            .forward(&x)
            .context("Forward pass failed at linear2")?;

        Ok(output)
    }

    fn parameters(&mut self) -> Vec<&mut dyn mini_burn::optim::Parameter<Backend>> {
        let mut params = Vec::new();
        // In a real implementation, we would collect parameters from layers
        // This is a simplified version for demonstration
        params
    }

    fn train(&mut self) {
        self.training_mode = true;
    }

    fn eval(&mut self) {
        self.training_mode = false;
    }
}

/// Demonstration of automatic differentiation
fn demo_autodiff() -> Result<()> {
    println!("=== 1. Autodiff Demonstration ===");

    // Create tensors that require gradients
    let shape = Shape::new([2, 2]);
    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![0.5, 1.5, 2.5, 3.5];

    let tensor1: Tensor<Backend, 2> = Tensor::from_data(data1, shape.clone());
    let tensor2: Tensor<Backend, 2> = Tensor::from_data(data2, shape);

    // Create gradient tensors
    let grad_tensor1 = GradTensor::with_grad(tensor1);
    let grad_tensor2 = GradTensor::with_grad(tensor2);

    println!("Created tensors with gradient tracking enabled");
    println!("Tensor 1 requires grad: {}", grad_tensor1.requires_grad());
    println!("Tensor 2 requires grad: {}", grad_tensor2.requires_grad());

    // Demonstrate gradient context
    with_grad(|| {
        println!("Inside gradient context - tracking enabled");

        no_grad(|| {
            println!("Inside no_grad context - tracking disabled");
        });

        println!("Back in gradient context");
    });

    println!("Autodiff demonstration completed!\n");
    Ok(())
}

/// Demonstration of optimizers
fn demo_optimizers() -> Result<()> {
    println!("=== 2. Optimizers Demonstration ===");

    // Create some dummy parameters
    let shape = Shape::new([3, 2]);
    let tensor_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let grad_data = vec![0.1, 0.2, 0.1, 0.3, 0.2, 0.1];

    let tensor: Tensor<Backend, 2> = Tensor::from_data(tensor_data, shape.clone());
    let grad: Tensor<Backend, 2> = Tensor::from_data(grad_data, shape);

    let mut grad_tensor = GradTensor::with_grad(tensor);
    grad_tensor.set_grad(grad);

    let mut param = TensorParameter::new(grad_tensor, "weight".to_string());

    // Demonstrate SGD optimizer
    println!("--- SGD Optimizer ---");
    let mut sgd = Sgd::with_momentum(0.01, 0.9);
    println!("SGD learning rate: {}", sgd.learning_rate());
    println!("SGD momentum: {}", sgd.momentum());

    let mut params: Vec<&mut dyn mini_burn::optim::Parameter<Backend>> = vec![&mut param];

    // Simulate optimization step
    println!("Before optimization step: {}", sgd.step_count());
    sgd.step(&mut params);
    println!("After optimization step: {}", sgd.step_count());

    // Demonstrate Adam optimizer
    println!("\n--- Adam Optimizer ---");
    let mut adam = Adam::with_betas(0.001, 0.9, 0.999)
        .with_weight_decay(0.01)
        .with_epsilon(1e-8);

    println!("Adam learning rate: {}", adam.learning_rate());
    println!("Adam beta1: {}", adam.beta1());
    println!("Adam beta2: {}", adam.beta2());
    println!("Adam epsilon: {}", adam.epsilon());

    adam.step(&mut params);
    println!("Adam step completed");

    println!("Optimizers demonstration completed!\n");
    Ok(())
}

/// Demonstration of training loop
fn demo_training_loop() -> Result<()> {
    println!("=== 3. Training Loop Demonstration ===");

    // Create training configuration
    let config = TrainingConfig::new(5, 0.001, 4)
        .with_accuracy()
        .with_validation_frequency(1);

    println!("Training configuration:");
    println!("  Epochs: {}", config.epochs);
    println!("  Learning rate: {}", config.learning_rate);
    println!("  Batch size: {}", config.batch_size);
    println!("  Compute accuracy: {}", config.compute_accuracy);

    // Create training loop with callback
    let mut training_loop = TrainingLoop::new(config);
    let logging_callback = LoggingCallback::new()
        .with_step_logging(2)
        .with_epoch_frequency(1);

    training_loop.add_callback(Box::new(logging_callback));

    // Simulate training metrics
    let mut metrics = TrainingMetrics::new();

    // Simulate some training epochs
    for epoch in 1..=3 {
        let train_loss = 1.0 / epoch as f32; // Decreasing loss
        let val_loss = train_loss + 0.1;
        let train_acc = 0.5 + (epoch as f32 * 0.15);
        let val_acc = train_acc - 0.05;

        metrics.add_train_loss(train_loss);
        metrics.add_val_loss(val_loss);
        metrics.add_train_accuracy(train_acc);
        metrics.add_val_accuracy(val_acc);

        let epoch_result = EpochResult::new(epoch, train_loss, 100)
            .with_validation(val_loss, 20)
            .with_train_accuracy(train_acc)
            .with_val_accuracy(val_acc)
            .with_duration(1.5);

        println!(
            "Epoch {}: train_loss={:.3}, val_loss={:.3}, train_acc={:.3}, val_acc={:.3}",
            epoch, train_loss, val_loss, train_acc, val_acc
        );
    }

    println!("Latest train loss: {:?}", metrics.latest_train_loss());
    println!("Best validation loss: {:?}", metrics.best_val_loss());
    println!(
        "Best validation accuracy: {:?}",
        metrics.best_val_accuracy()
    );

    println!("Training loop demonstration completed!\n");
    Ok(())
}

/// Demonstration of broadcasting
fn demo_broadcasting() -> Result<()> {
    println!("=== 4. Broadcasting Demonstration ===");

    // Test compatible shapes
    let shape1 = vec![2, 3];
    let shape2 = vec![1, 3];
    let shape3 = vec![2, 1];

    println!("Shape compatibility tests:");
    println!(
        "  {:?} + {:?} = compatible: {}",
        shape1,
        shape2,
        Broadcasting::are_compatible(&shape1, &shape2)
    );
    println!(
        "  {:?} + {:?} = compatible: {}",
        shape1,
        shape3,
        Broadcasting::are_compatible(&shape1, &shape3)
    );

    // Compute broadcast shapes
    match Broadcasting::broadcast_shape(&shape1, &shape2) {
        Ok(result) => println!(
            "  Broadcast shape of {:?} and {:?} = {:?}",
            shape1, shape2, result
        ),
        Err(e) => println!("  Broadcasting failed: {}", e),
    }

    // Test incompatible shapes
    let incompatible1 = vec![2, 3];
    let incompatible2 = vec![2, 4];

    println!("\nIncompatible shapes test:");
    match Broadcasting::broadcast_shape(&incompatible1, &incompatible2) {
        Ok(result) => println!("  Unexpected success: {:?}", result),
        Err(e) => println!("  Expected failure: {}", e),
    }

    // Demonstrate broadcast strides
    let original = vec![3];
    let target = vec![2, 3];

    match Broadcasting::broadcast_strides(&original, &target) {
        Ok(strides) => println!(
            "  Broadcast strides from {:?} to {:?} = {:?}",
            original, target, strides
        ),
        Err(e) => println!("  Stride computation failed: {}", e),
    }

    // Demonstrate shape operations
    let shape = vec![1, 4, 1, 3, 1];
    let squeezed = Broadcasting::squeeze_shape(&shape);
    println!("  Squeezed shape of {:?} = {:?}", shape, squeezed);

    let unsqueezed =
        Broadcasting::unsqueeze_shape(&vec![3, 4], 1).context("Failed to unsqueeze shape")?;
    println!("  Unsqueezed [3, 4] at dim 1 = {:?}", unsqueezed);

    println!("Broadcasting demonstration completed!\n");
    Ok(())
}

/// Demonstration of error handling
fn demo_error_handling() -> Result<()> {
    println!("=== 5. Error Handling Demonstration ===");

    // Demonstrate different error types
    println!("--- Error Type Examples ---");

    // Tensor error
    let tensor_error = mini_burn::error::TensorError::ShapeMismatch {
        lhs: vec![2, 3],
        rhs: vec![3, 4],
        operation: "matmul".to_string(),
    };
    println!("Tensor error: {}", tensor_error);

    // Shape error
    let shape_error = mini_burn::error::ShapeError::CannotReshape {
        from: vec![2, 3],
        to: vec![5],
    };
    println!("Shape error: {}", shape_error);

    // Memory error
    let memory_error = mini_burn::error::MemoryError::OutOfMemory {
        requested: 1024 * 1024,
        available: 512 * 1024,
    };
    println!("Memory error: {}", memory_error);

    // Demonstrate error context
    println!("\n--- Error Context Example ---");
    let result: std::result::Result<(), mini_burn::error::TensorError> =
        Err(mini_burn::error::TensorError::DivisionByZero);

    match result.context("During neural network forward pass") {
        Ok(_) => println!("Unexpected success"),
        Err(e) => println!("Error with context: {}", e),
    }

    // Demonstrate error recovery
    println!("\n--- Error Recovery Example ---");
    match create_tensor_with_validation(vec![2, 0, 3]) {
        Ok(shape) => println!("Valid shape created: {:?}", shape),
        Err(e) => {
            println!("Caught error: {}", e);
            println!("Attempting recovery with default shape...");

            // Recovery: use default shape
            let default_shape = vec![2, 2];
            println!("Using default shape: {:?}", default_shape);
        }
    }

    // Demonstrate chained errors
    println!("\n--- Error Chain Example ---");
    match perform_complex_operation() {
        Ok(_) => println!("Operation succeeded"),
        Err(e) => {
            println!("Main error: {}", e);

            // Walk the error chain
            let mut source = e.source();
            let mut level = 1;
            while let Some(err) = source {
                println!("  Caused by (level {}): {}", level, err);
                source = err.source();
                level += 1;
            }
        }
    }

    println!("Error handling demonstration completed!\n");
    Ok(())
}

/// Helper function to demonstrate error validation
fn create_tensor_with_validation(shape: Vec<usize>) -> Result<Vec<usize>> {
    // Check for zero dimensions
    for (i, &dim) in shape.iter().enumerate() {
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

/// Helper function to demonstrate error chaining
fn perform_complex_operation() -> Result<()> {
    // Simulate a chain of operations that can fail
    validate_input().context("Input validation phase")?;

    process_data().context("Data processing phase")?;

    finalize_operation().context("Finalization phase")?;

    Ok(())
}

fn validate_input() -> Result<()> {
    Err(MiniBurnError::Config(
        mini_burn::error::ConfigError::InvalidValue {
            key: "batch_size".to_string(),
            value: "0".to_string(),
            reason: "must be positive".to_string(),
        },
    ))
}

fn process_data() -> Result<()> {
    Ok(())
}

fn finalize_operation() -> Result<()> {
    Ok(())
}

/// Main function demonstrating all core features
fn main() -> Result<()> {
    println!("Mini-Burn Core Features Demonstration");
    println!("=====================================\n");

    // 1. Automatic Differentiation
    demo_autodiff().context("Autodiff demonstration failed")?;

    // 2. Optimizers
    demo_optimizers().context("Optimizers demonstration failed")?;

    // 3. Training Loop
    demo_training_loop().context("Training loop demonstration failed")?;

    // 4. Broadcasting
    demo_broadcasting().context("Broadcasting demonstration failed")?;

    // 5. Error Handling
    demo_error_handling().context("Error handling demonstration failed")?;

    println!("=== Summary ===");
    println!("✅ 1. Autodiff: Implemented with gradient tracking and computation graphs");
    println!("✅ 2. Optimizers: SGD and Adam with momentum, weight decay, and scheduling");
    println!("✅ 3. Training Loop: Automated training with epochs, validation, and callbacks");
    println!("✅ 4. Broadcasting: NumPy-style broadcasting for tensor operations");
    println!("✅ 5. Error Handling: Comprehensive error types with context and recovery");

    println!("\n🎉 Mini-Burn framework now has all 5 core features for a USABLE deep learning framework!");
    println!("\nNext steps for COMPETITIVE framework:");
    println!("  6. GPU support");
    println!("  7. Model serialization");
    println!("  8. Convolutional layers");
    println!("  9. Memory optimization");
    println!(" 10. Data loading utilities");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_net_creation() {
        let result = SimpleNet::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_error_validation() {
        // Valid shape should pass
        let valid_result = create_tensor_with_validation(vec![2, 3, 4]);
        assert!(valid_result.is_ok());

        // Zero dimension should fail
        let invalid_result = create_tensor_with_validation(vec![2, 0, 4]);
        assert!(invalid_result.is_err());

        // Too many dimensions should fail
        let too_many_dims = vec![1; 10];
        let invalid_result = create_tensor_with_validation(too_many_dims);
        assert!(invalid_result.is_err());
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
}
