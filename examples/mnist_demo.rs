//! MNIST Demo - Computer Vision with Mini-Burn Framework
//!
//! This example demonstrates how to use Mini-Burn for computer vision tasks
//! similar to MNIST digit classification. Since we don't have external dependencies
//! for data loading, we simulate MNIST-like data and showcase:
//!
//! - Neural network architecture for image classification
//! - Batch processing of images
//! - Multi-layer networks with appropriate activations
//! - Classification metrics and evaluation
//! - Realistic forward pass through CNN-like architecture

use mini_burn::backend::CpuBackend;
use mini_burn::shape::Shape;
use mini_burn::tensor::Tensor;
use std::collections::HashMap;

/// MNIST-like constants
const IMAGE_SIZE: usize = 28; // 28x28 pixels
const NUM_PIXELS: usize = IMAGE_SIZE * IMAGE_SIZE; // 784 pixels
const NUM_CLASSES: usize = 10; // Digits 0-9
const BATCH_SIZE: usize = 8;

/// Linear layer implementation for our neural network
struct LinearLayer {
    weights: Tensor<CpuBackend, 2>,
    bias: Tensor<CpuBackend, 1>,
    input_size: usize,
    output_size: usize,
}

impl LinearLayer {
    /// Initialize a new linear layer with Xavier-like initialization
    fn new(input_size: usize, output_size: usize) -> Self {
        // Xavier initialization: weights ~ U(-sqrt(6/(input+output)), sqrt(6/(input+output)))
        let bound = (6.0 / (input_size + output_size) as f32).sqrt();

        let weight_data: Vec<f32> = (0..input_size * output_size)
            .map(|i| {
                let val = (i as f32 * 0.1) % 2.0 - 1.0; // Simple pseudo-random
                val * bound
            })
            .collect();

        let bias_data: Vec<f32> = (0..output_size)
            .map(|i| (i as f32 * 0.05) % 0.1 - 0.05)
            .collect();

        let weights = Tensor::from_data(weight_data, Shape::new([input_size, output_size]));
        let bias = Tensor::from_data(bias_data, Shape::new([output_size]));

        Self {
            weights,
            bias,
            input_size,
            output_size,
        }
    }

    /// Forward pass for batch of inputs
    fn forward_batch(&self, input: &Tensor<CpuBackend, 2>) -> Tensor<CpuBackend, 2> {
        // input shape: [batch_size, input_size]
        // weights shape: [input_size, output_size]
        // output shape: [batch_size, output_size]

        let linear_output = input.matmul(&self.weights);

        // Add bias to each sample in the batch
        let linear_data = linear_output.to_data();
        let bias_data = self.bias.to_data();
        let batch_size = input.shape().dim(0);

        let mut output_data = vec![0.0; batch_size * self.output_size];

        for batch_idx in 0..batch_size {
            for feature_idx in 0..self.output_size {
                let data_idx = batch_idx * self.output_size + feature_idx;
                output_data[data_idx] = linear_data[data_idx] + bias_data[feature_idx];
            }
        }

        Tensor::from_data(output_data, Shape::new([batch_size, self.output_size]))
    }
}

/// MNIST-like Neural Network Architecture
struct MNISTNet {
    // Flatten layer (implicit: 28x28 -> 784)
    fc1: LinearLayer, // 784 -> 128
    fc2: LinearLayer, // 128 -> 64
    fc3: LinearLayer, // 64 -> 32
    fc4: LinearLayer, // 32 -> 10 (output classes)
}

impl MNISTNet {
    fn new() -> Self {
        Self {
            fc1: LinearLayer::new(NUM_PIXELS, 128),
            fc2: LinearLayer::new(128, 64),
            fc3: LinearLayer::new(64, 32),
            fc4: LinearLayer::new(32, NUM_CLASSES),
        }
    }

    /// Forward pass through the network
    fn forward(&self, input: &Tensor<CpuBackend, 2>) -> Tensor<CpuBackend, 2> {
        // Layer 1: Linear + ReLU
        let x1 = self.fc1.forward_batch(input);
        let x1_activated = x1.relu();

        // Layer 2: Linear + ReLU
        let x2 = self.fc2.forward_batch(&x1_activated);
        let x2_activated = x2.relu();

        // Layer 3: Linear + ReLU
        let x3 = self.fc3.forward_batch(&x2_activated);
        let x3_activated = x3.relu();

        // Output layer: Linear + Softmax
        let x4 = self.fc4.forward_batch(&x3_activated);
        x4.softmax() // Softmax for classification probabilities
    }

    /// Predict single image
    fn predict_single(&self, image: &Tensor<CpuBackend, 1>) -> (usize, f32) {
        // Convert 1D image to 2D batch format
        let batch_input = Tensor::from_data(image.to_data(), Shape::new([1, NUM_PIXELS]));

        let output = self.forward(&batch_input);
        let probabilities = output.to_data();

        // Find class with highest probability
        let (predicted_class, confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &prob)| (idx, prob))
            .unwrap();

        (predicted_class, confidence)
    }
}

/// Generate synthetic MNIST-like data for demonstration
fn generate_synthetic_mnist() -> (Vec<Tensor<CpuBackend, 1>>, Vec<usize>) {
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for digit in 0..NUM_CLASSES {
        // Generate 3 samples per digit
        for sample in 0..3 {
            let mut image_data = vec![0.0; NUM_PIXELS];

            // Create simple patterns for each digit
            match digit {
                0 => generate_circle_pattern(&mut image_data),
                1 => generate_vertical_line_pattern(&mut image_data),
                2 => generate_horizontal_lines_pattern(&mut image_data),
                3 => generate_curves_pattern(&mut image_data),
                4 => generate_triangle_pattern(&mut image_data),
                5 => generate_square_pattern(&mut image_data),
                6 => generate_oval_pattern(&mut image_data),
                7 => generate_diagonal_pattern(&mut image_data),
                8 => generate_cross_pattern(&mut image_data),
                9 => generate_spiral_pattern(&mut image_data),
                _ => {}
            }

            // Add some noise based on sample number
            let noise_level = sample as f32 * 0.1;
            for pixel in image_data.iter_mut() {
                *pixel += (sample as f32 * 0.03) % 0.2 - 0.1; // Add variation
                *pixel = pixel.max(0.0).min(1.0); // Clamp to [0, 1]
            }

            let image = Tensor::from_data(image_data, Shape::new([NUM_PIXELS]));
            images.push(image);
            labels.push(digit);
        }
    }

    (images, labels)
}

/// Generate circle pattern (digit 0)
fn generate_circle_pattern(data: &mut [f32]) {
    let center_x = IMAGE_SIZE / 2;
    let center_y = IMAGE_SIZE / 2;
    let radius = 8.0;

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let dx = (x as f32 - center_x as f32);
            let dy = (y as f32 - center_y as f32);
            let distance = (dx * dx + dy * dy).sqrt();

            if (distance - radius).abs() < 2.0 {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate vertical line pattern (digit 1)
fn generate_vertical_line_pattern(data: &mut [f32]) {
    let center_x = IMAGE_SIZE / 2;

    for y in 5..IMAGE_SIZE - 5 {
        for x in center_x - 1..center_x + 2 {
            if x < IMAGE_SIZE {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate horizontal lines pattern (digit 2)
fn generate_horizontal_lines_pattern(data: &mut [f32]) {
    let lines = [7, 14, 21];

    for &line_y in &lines {
        for x in 5..IMAGE_SIZE - 5 {
            data[line_y * IMAGE_SIZE + x] = 1.0;
        }
    }
}

/// Generate curves pattern (digit 3)
fn generate_curves_pattern(data: &mut [f32]) {
    for y in 0..IMAGE_SIZE {
        let x1 = (10.0 + 5.0 * (y as f32 * 0.3).sin()) as usize;
        let x2 = (18.0 + 5.0 * (y as f32 * 0.3).cos()) as usize;

        if x1 < IMAGE_SIZE {
            data[y * IMAGE_SIZE + x1] = 1.0;
        }
        if x2 < IMAGE_SIZE {
            data[y * IMAGE_SIZE + x2] = 1.0;
        }
    }
}

/// Generate triangle pattern (digit 4)
fn generate_triangle_pattern(data: &mut [f32]) {
    for y in 5..IMAGE_SIZE - 5 {
        for x in 5..IMAGE_SIZE - 5 {
            if x == 10 || y == 15 || (x + y == 25 && x < 15) {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate square pattern (digit 5)
fn generate_square_pattern(data: &mut [f32]) {
    for y in 8..20 {
        for x in 8..20 {
            if y == 8 || y == 19 || x == 8 || x == 19 {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate oval pattern (digit 6)
fn generate_oval_pattern(data: &mut [f32]) {
    let center_x = 14.0;
    let center_y = 14.0;

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let dx = (x as f32 - center_x) / 6.0;
            let dy = (y as f32 - center_y) / 9.0;

            if (dx * dx + dy * dy - 1.0).abs() < 0.3 {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate diagonal pattern (digit 7)
fn generate_diagonal_pattern(data: &mut [f32]) {
    for y in 5..IMAGE_SIZE - 5 {
        let x = (5 + y) % (IMAGE_SIZE - 10) + 5;
        if x < IMAGE_SIZE {
            data[y * IMAGE_SIZE + x] = 1.0;
            if x > 0 {
                data[y * IMAGE_SIZE + x - 1] = 0.7;
            }
            if x < IMAGE_SIZE - 1 {
                data[y * IMAGE_SIZE + x + 1] = 0.7;
            }
        }
    }
}

/// Generate cross pattern (digit 8)
fn generate_cross_pattern(data: &mut [f32]) {
    let center = IMAGE_SIZE / 2;

    // Vertical line
    for y in 5..IMAGE_SIZE - 5 {
        data[y * IMAGE_SIZE + center] = 1.0;
    }

    // Horizontal line
    for x in 5..IMAGE_SIZE - 5 {
        data[center * IMAGE_SIZE + x] = 1.0;
    }
}

/// Generate spiral pattern (digit 9)
fn generate_spiral_pattern(data: &mut [f32]) {
    let center_x = 14.0;
    let center_y = 14.0;

    for i in 0..100 {
        let angle = i as f32 * 0.3;
        let radius = i as f32 * 0.1;

        let x = (center_x + radius * angle.cos()) as usize;
        let y = (center_y + radius * angle.sin()) as usize;

        if x < IMAGE_SIZE && y < IMAGE_SIZE {
            data[y * IMAGE_SIZE + x] = 1.0;
        }
    }
}

/// Evaluation metrics
struct Metrics {
    correct: usize,
    total: usize,
    class_correct: HashMap<usize, usize>,
    class_total: HashMap<usize, usize>,
}

impl Metrics {
    fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
            class_correct: HashMap::new(),
            class_total: HashMap::new(),
        }
    }

    fn add_prediction(&mut self, predicted: usize, actual: usize) {
        self.total += 1;

        let class_total = self.class_total.entry(actual).or_insert(0);
        *class_total += 1;

        if predicted == actual {
            self.correct += 1;
            let class_correct = self.class_correct.entry(actual).or_insert(0);
            *class_correct += 1;
        }
    }

    fn accuracy(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f32 / self.total as f32
        }
    }

    fn class_accuracy(&self, class: usize) -> f32 {
        let correct = self.class_correct.get(&class).unwrap_or(&0);
        let total = self.class_total.get(&class).unwrap_or(&0);
        if *total == 0 {
            0.0
        } else {
            *correct as f32 / *total as f32
        }
    }
}

/// Visualize a 28x28 image in ASCII
fn print_image(image: &Tensor<CpuBackend, 1>) {
    let data = image.to_data();

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let pixel = data[y * IMAGE_SIZE + x];
            let char = if pixel > 0.7 {
                '█'
            } else if pixel > 0.4 {
                '▓'
            } else if pixel > 0.1 {
                '░'
            } else {
                ' '
            };
            print!("{}", char);
        }
        println!();
    }
}

fn main() {
    println!("=== MNIST Demo with Mini-Burn Framework ===\n");

    // 1. Generate synthetic MNIST-like dataset
    println!("1. Generating Synthetic MNIST-like Dataset");
    let (images, labels) = generate_synthetic_mnist();
    println!(
        "Generated {} images with {} classes",
        images.len(),
        NUM_CLASSES
    );
    println!(
        "Image dimensions: {}x{} = {} pixels\n",
        IMAGE_SIZE, IMAGE_SIZE, NUM_PIXELS
    );

    // 2. Initialize neural network
    println!("2. Initializing Neural Network");
    let model = MNISTNet::new();
    println!(
        "Architecture: {} -> 128 -> 64 -> 32 -> {}",
        NUM_PIXELS, NUM_CLASSES
    );
    println!("Activation functions: ReLU (hidden) + Softmax (output)\n");

    // 3. Visualize sample images
    println!("3. Sample Images Visualization");
    for digit in 0..NUM_CLASSES {
        println!("Digit {} sample:", digit);
        print_image(&images[digit * 3]); // Show first sample of each digit
        println!();
    }

    // 4. Single image inference
    println!("4. Single Image Inference Examples");
    for i in 0..10 {
        let (predicted, confidence) = model.predict_single(&images[i]);
        let actual = labels[i];
        println!(
            "Image {}: Predicted = {}, Actual = {}, Confidence = {:.3}, {}",
            i + 1,
            predicted,
            actual,
            confidence,
            if predicted == actual { "✓" } else { "✗" }
        );
    }
    println!();

    // 5. Batch processing
    println!("5. Batch Processing Demo");

    // Create a batch of images
    let batch_indices: Vec<usize> = (0..BATCH_SIZE).collect();
    let mut batch_data = Vec::new();

    for &idx in &batch_indices {
        batch_data.extend(images[idx].to_data());
    }

    let batch_input = Tensor::from_data(batch_data, Shape::new([BATCH_SIZE, NUM_PIXELS]));
    let batch_output = model.forward(&batch_input);
    let batch_probs = batch_output.to_data();

    println!("Batch size: {}", BATCH_SIZE);
    println!("Input shape: {:?}", batch_input.shape().dims());
    println!("Output shape: {:?}", batch_output.shape().dims());

    for i in 0..BATCH_SIZE {
        let start_idx = i * NUM_CLASSES;
        let end_idx = start_idx + NUM_CLASSES;
        let sample_probs = &batch_probs[start_idx..end_idx];

        let (predicted_class, confidence) = sample_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, &prob)| (idx, prob))
            .unwrap();

        println!(
            "Sample {}: Predicted = {}, Actual = {}, Confidence = {:.3}",
            i + 1,
            predicted_class,
            labels[batch_indices[i]],
            confidence
        );
    }
    println!();

    // 6. Full dataset evaluation
    println!("6. Full Dataset Evaluation");
    let mut metrics = Metrics::new();

    for (i, (image, &actual_label)) in images.iter().zip(labels.iter()).enumerate() {
        let (predicted_label, confidence) = model.predict_single(image);
        metrics.add_prediction(predicted_label, actual_label);

        if i < 5 {
            // Show first 5 detailed predictions
            println!(
                "Sample {}: Predicted = {}, Actual = {}, Confidence = {:.3}",
                i + 1,
                predicted_label,
                actual_label,
                confidence
            );
        }
    }

    println!("\n=== Evaluation Results ===");
    println!(
        "Overall Accuracy: {:.1}% ({}/{})",
        metrics.accuracy() * 100.0,
        metrics.correct,
        metrics.total
    );

    println!("\nPer-class Accuracy:");
    for class in 0..NUM_CLASSES {
        println!(
            "  Digit {}: {:.1}%",
            class,
            metrics.class_accuracy(class) * 100.0
        );
    }

    // 7. Probability distribution analysis
    println!("\n7. Probability Distribution Analysis");
    let sample_image = &images[0]; // Analyze first image (digit 0)
    let (predicted, _) = model.predict_single(sample_image);

    // Get full probability distribution
    let batch_input = Tensor::from_data(sample_image.to_data(), Shape::new([1, NUM_PIXELS]));
    let output = model.forward(&batch_input);
    let probabilities = output.to_data();

    println!("Sample image (Digit {}):", labels[0]);
    println!("Probability distribution:");
    for (class, &prob) in probabilities.iter().enumerate() {
        let bar_length = (prob * 50.0) as usize;
        let bar = "█".repeat(bar_length);
        println!("  {}: {:.3} {}", class, prob, bar);
    }

    // 8. Network layer analysis
    println!("\n8. Network Layer Analysis");
    println!("Layer sizes:");
    println!(
        "  Input layer: {} neurons (flattened 28x28 image)",
        NUM_PIXELS
    );
    println!("  Hidden layer 1: 128 neurons (ReLU activation)");
    println!("  Hidden layer 2: 64 neurons (ReLU activation)");
    println!("  Hidden layer 3: 32 neurons (ReLU activation)");
    println!(
        "  Output layer: {} neurons (Softmax activation)",
        NUM_CLASSES
    );

    let total_params =
        NUM_PIXELS * 128 + 128 + 128 * 64 + 64 + 64 * 32 + 32 + 32 * NUM_CLASSES + NUM_CLASSES;
    println!("Total parameters: {}", total_params);

    // 9. Activation function comparison
    println!("\n9. Activation Function Effects");
    let test_input: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new([5]));

    println!("Test input: {:?}", test_input.to_data());
    println!("ReLU output: {:?}", test_input.relu().to_data());
    println!("Sigmoid output: {:?}", test_input.sigmoid().to_data());
    println!("Tanh output: {:?}", test_input.tanh().to_data());

    // 10. Performance characteristics
    println!("\n10. Performance Characteristics");
    println!("Forward pass operations per image:");
    println!("  Matrix multiplications: 4");
    println!("  Activation functions: 4 (3 ReLU + 1 Softmax)");
    println!(
        "  Total FLOPs per image: ~{}",
        2 * (NUM_PIXELS * 128 + 128 * 64 + 64 * 32 + 32 * NUM_CLASSES)
    );

    println!("\n=== MNIST Demo Complete ===");
    println!("This demonstrates:");
    println!("✓ Computer vision neural network architecture");
    println!("✓ Batch processing of images");
    println!("✓ Multi-class classification with softmax");
    println!("✓ Activation functions (ReLU, Softmax)");
    println!("✓ Model evaluation and metrics");
    println!("✓ Synthetic dataset generation");
    println!("✓ ASCII image visualization");
    println!("✓ Probability distribution analysis");

    println!("\nNext steps for real MNIST:");
    println!("- Load actual MNIST dataset");
    println!("- Implement automatic differentiation for training");
    println!("- Add convolutional layers");
    println!("- Implement optimizers (SGD, Adam)");
    println!("- Add data augmentation");
}
