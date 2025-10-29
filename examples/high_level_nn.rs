//! High-Level Neural Network Example using Mini-Burn Framework
//!
//! This example demonstrates how to use the high-level neural network APIs
//! provided by Mini-Burn framework to build and use models easily.

use mini_burn::backend::CpuBackend;
use mini_burn::nn::{
    layers::{Linear, ReLU, Softmax, Swish, GELU},
    loss::{CrossEntropyLoss, Loss, LossUtils, MSELoss},
    ModelBuilder, Sequential, Trainer,
};
use mini_burn::shape::Shape;
use mini_burn::tensor::Tensor;

fn main() {
    println!("=== High-Level Neural Network API Demo ===\n");

    // 1. Build models using ModelBuilder
    demonstrate_model_builder();

    // 2. Build models using Sequential API
    demonstrate_sequential_api();

    // 3. Loss functions demonstration
    demonstrate_loss_functions();

    // 4. Training utilities
    demonstrate_training_utilities();

    // 5. Real classification example
    demonstrate_classification_example();
}

fn demonstrate_model_builder() {
    println!("1. Model Builder Demo");

    // Build a classifier automatically
    let classifier = ModelBuilder::<CpuBackend>::classifier(
        784,        // input features (28x28 image)
        &[128, 64], // hidden layer sizes
        10,         // number of classes
    );

    println!("Created classifier with ModelBuilder:");
    println!("  Architecture: 784 → 128 → 64 → 10");
    println!("  Layers: {}", classifier.len());

    // Build an MLP
    let mlp = ModelBuilder::<CpuBackend>::mlp(&[100, 50, 25, 10]);
    println!("Created MLP: 100 → 50 → 25 → 10");
    println!("  Layers: {}\n", mlp.len());
}

fn demonstrate_sequential_api() {
    println!("2. Sequential API Demo");

    // Build model layer by layer
    let model = Sequential::<CpuBackend>::new()
        .add(Linear::new(784, 256))
        .add(ReLU::new())
        .add(Linear::new(256, 128))
        .add(GELU::new())
        .add(Linear::new(128, 64))
        .add(Swish::new())
        .add(Linear::new(64, 10))
        .add(Softmax::new());

    println!("Built sequential model with mixed activations:");
    println!("  784 → Linear → ReLU");
    println!("  → 256 → Linear → GELU");
    println!("  → 128 → Linear → Swish");
    println!("  → 64 → Linear → Softmax → 10");

    // Test forward pass
    let input = Tensor::from_data(vec![1.0; 784], Shape::new([784]));
    let output = model.predict(&input);

    println!("Forward pass test:");
    println!("  Input shape: {:?}", input.shape().dims());
    println!("  Output shape: {:?}", output.shape().dims());

    let probabilities = output.to_data();
    let sum: f32 = probabilities.iter().sum();
    println!("  Probability sum: {:.6} (should be ~1.0)", sum);

    // Show top 3 predictions
    let mut indexed_probs: Vec<(usize, f32)> = probabilities
        .iter()
        .enumerate()
        .map(|(i, &p)| (i, p))
        .collect();
    indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("  Top 3 predictions:");
    for (i, (class, prob)) in indexed_probs.iter().take(3).enumerate() {
        println!("    {}: Class {} ({:.3})", i + 1, class, prob);
    }
    println!();
}

fn demonstrate_loss_functions() {
    println!("3. Loss Functions Demo");

    // Create some dummy predictions and targets
    let predictions = Tensor::from_data(
        vec![
            0.9, 0.1, 0.0, // Sample 1: confident about class 0
            0.2, 0.7, 0.1, // Sample 2: confident about class 1
            0.1, 0.2, 0.7, // Sample 3: confident about class 2
        ],
        Shape::new([3, 3]),
    );

    let targets = vec![0, 1, 2]; // Ground truth classes

    // Cross Entropy Loss
    let ce_loss = CrossEntropyLoss::<CpuBackend>::new();
    let ce_value = ce_loss.compute_with_indices(&predictions, &targets);
    println!("Cross Entropy Loss: {:.4}", ce_value);

    // Convert to one-hot for full CE computation
    let one_hot_targets = LossUtils::indices_to_one_hot::<CpuBackend>(&targets, 3);
    let ce_full = ce_loss.compute(&predictions, &one_hot_targets);
    println!("Cross Entropy (one-hot): {:.4}", ce_full);

    // MSE Loss (for regression example)
    let regression_pred = Tensor::from_data(vec![1.2, 2.8, 3.1, 4.9], Shape::new([2, 2]));
    let regression_target = Tensor::from_data(vec![1.0, 3.0, 3.0, 5.0], Shape::new([2, 2]));

    let mse_loss = MSELoss::<CpuBackend>::new();
    let mse_value = mse_loss.compute(&regression_pred, &regression_target);
    println!("MSE Loss: {:.4}", mse_value);

    // Accuracy computation
    let accuracy = LossUtils::accuracy(&predictions, &targets);
    println!("Accuracy: {:.1}%\n", accuracy * 100.0);
}

fn demonstrate_training_utilities() {
    println!("4. Training Utilities Demo");

    // Create a simple model
    let model = ModelBuilder::<CpuBackend>::classifier(4, &[8, 4], 2);

    // Create dummy dataset
    let images = vec![
        Tensor::from_data(vec![1.0, 0.0, 0.0, 0.0], Shape::new([4])), // Class 0
        Tensor::from_data(vec![0.0, 1.0, 0.0, 0.0], Shape::new([4])), // Class 1
        Tensor::from_data(vec![1.0, 1.0, 0.0, 0.0], Shape::new([4])), // Class 0
        Tensor::from_data(vec![0.0, 0.0, 1.0, 1.0], Shape::new([4])), // Class 1
        Tensor::from_data(vec![0.8, 0.2, 0.0, 0.0], Shape::new([4])), // Class 0
        Tensor::from_data(vec![0.2, 0.8, 0.1, 0.1], Shape::new([4])), // Class 1
    ];

    let labels = vec![0, 1, 0, 1, 0, 1];

    // Evaluate model
    let (overall_acc, class_accs) = Trainer::evaluate(&model, &images, &labels);

    println!("Model evaluation on dummy dataset:");
    println!("  Overall accuracy: {:.1}%", overall_acc * 100.0);
    println!("  Class accuracies:");
    for (i, &acc) in class_accs.iter().enumerate() {
        println!("    Class {}: {:.1}%", i, acc * 100.0);
    }

    // Batch prediction
    let batch_data = vec![
        1.0, 0.0, 0.0, 0.0, // Sample 1
        0.0, 1.0, 0.0, 0.0, // Sample 2
        0.5, 0.5, 0.0, 0.0, // Sample 3
    ];
    let batch = Tensor::from_data(batch_data, Shape::new([3, 4]));

    let predictions = Trainer::batch_predict(&model, &batch);
    println!("  Batch predictions:");
    for (i, (class, confidence)) in predictions.iter().enumerate() {
        println!("    Sample {}: Class {} ({:.3})", i + 1, class, confidence);
    }
    println!();
}

fn demonstrate_classification_example() {
    println!("5. Classification Example");

    // Create a 3-class classification problem
    let model = Sequential::<CpuBackend>::new()
        .add(Linear::new(8, 16))
        .add(ReLU::new())
        .add(Linear::new(16, 8))
        .add(ReLU::new())
        .add(Linear::new(8, 3))
        .add(Softmax::new());

    println!("Created 3-class classifier: 8 → 16 → 8 → 3");

    // Generate synthetic patterns for classification
    let (dataset, labels, class_names) = create_synthetic_dataset();

    println!("Generated synthetic dataset:");
    println!("  {} samples", dataset.len());
    println!("  Classes: {:?}", class_names);

    // Show some samples
    println!("Sample data:");
    for i in 0..3 {
        let data = dataset[i].to_data();
        println!(
            "  {} {}: [{:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}, {:.2}]",
            class_names[labels[i]],
            i + 1,
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
            data[5],
            data[6],
            data[7]
        );
    }

    // Evaluate model
    let (accuracy, class_accuracies) = Trainer::evaluate(&model, &dataset, &labels);

    println!("\nModel Performance:");
    println!("  Overall Accuracy: {:.1}%", accuracy * 100.0);
    println!("  Per-class Accuracy:");
    for (i, &acc) in class_accuracies.iter().enumerate() {
        println!("    {}: {:.1}%", class_names[i], acc * 100.0);
    }

    // Detailed predictions for first few samples
    println!("\nDetailed Predictions:");
    for i in 0..6 {
        let output = model.predict(&dataset[i]);
        let probs = output.to_data();

        let (predicted_class, &max_prob) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let actual_class = labels[i];
        let status = if predicted_class == actual_class {
            "✓"
        } else {
            "✗"
        };

        println!(
            "  Sample {}: {} → {} ({:.1}%) {}",
            i + 1,
            class_names[actual_class],
            class_names[predicted_class],
            max_prob * 100.0,
            status
        );

        // Show probability distribution
        println!("    Probabilities: [");
        for (j, (&prob, name)) in probs.iter().zip(class_names.iter()).enumerate() {
            let marker = if j == predicted_class { " ←" } else { "" };
            println!("      {}: {:.3}{}", name, prob, marker);
        }
        println!("    ]");
    }

    // Loss computation
    let ce_loss = CrossEntropyLoss::<CpuBackend>::new();
    let mut total_loss = 0.0;
    let mut total_samples = 0;

    for (image, &label) in dataset.iter().zip(labels.iter()) {
        let output = model.predict(image);
        let pred_batch = Tensor::from_data(output.to_data(), Shape::new([1, 3]));
        let loss = ce_loss.compute_with_indices(&pred_batch, &[label]);
        total_loss += loss;
        total_samples += 1;
    }

    let avg_loss = total_loss / total_samples as f32;
    println!("\nAverage Cross Entropy Loss: {:.4}", avg_loss);

    println!("\n=== High-Level API Demo Complete ===");
    println!("Demonstrated features:");
    println!("✓ ModelBuilder for quick model creation");
    println!("✓ Sequential API for custom architectures");
    println!("✓ Multiple activation functions (ReLU, GELU, Swish, Softmax)");
    println!("✓ Loss functions (CrossEntropy, MSE)");
    println!("✓ Training utilities (evaluation, batch prediction)");
    println!("✓ High-level classification workflow");
    println!("✓ Easy-to-use APIs that hide complexity");
}

/// Create synthetic dataset for classification demonstration
fn create_synthetic_dataset() -> (Vec<Tensor<CpuBackend, 1>>, Vec<usize>, Vec<String>) {
    let mut dataset = Vec::new();
    let mut labels = Vec::new();
    let class_names = vec![
        "TypeA".to_string(),
        "TypeB".to_string(),
        "TypeC".to_string(),
    ];

    // Generate patterns for each class
    for class in 0..3 {
        for sample in 0..4 {
            let mut features = vec![0.0; 8];

            match class {
                0 => {
                    // TypeA: High values in first half
                    features[0] = 0.8 + sample as f32 * 0.05;
                    features[1] = 0.7 + sample as f32 * 0.03;
                    features[2] = 0.6 + sample as f32 * 0.04;
                    features[3] = 0.5 + sample as f32 * 0.02;
                    features[4] = 0.1 + sample as f32 * 0.01;
                    features[5] = 0.2 + sample as f32 * 0.01;
                    features[6] = 0.1;
                    features[7] = 0.1;
                }
                1 => {
                    // TypeB: High values in second half
                    features[0] = 0.1;
                    features[1] = 0.2 + sample as f32 * 0.01;
                    features[2] = 0.1;
                    features[3] = 0.2;
                    features[4] = 0.6 + sample as f32 * 0.04;
                    features[5] = 0.7 + sample as f32 * 0.03;
                    features[6] = 0.8 + sample as f32 * 0.05;
                    features[7] = 0.9 + sample as f32 * 0.02;
                }
                2 => {
                    // TypeC: Balanced values
                    for i in 0..8 {
                        features[i] = 0.4 + sample as f32 * 0.02 + (i % 2) as f32 * 0.1;
                    }
                }
                _ => unreachable!(),
            }

            dataset.push(Tensor::from_data(features, Shape::new([8])));
            labels.push(class);
        }
    }

    (dataset, labels, class_names)
}
