//! Simple MNIST Classification Example
//!
//! This is a simplified version of MNIST digit classification that demonstrates
//! the core concepts of computer vision with Mini-Burn framework.
//!
//! Features:
//! - Simple neural network with better weight initialization
//! - Clear demonstration of forward pass
//! - Easy-to-understand synthetic data generation
//! - Step-by-step explanation of the classification process

use mini_burn::backend::CpuBackend;
use mini_burn::shape::Shape;
use mini_burn::tensor::Tensor;

const IMAGE_SIZE: usize = 8; // Simplified 8x8 images for faster demo
const NUM_PIXELS: usize = IMAGE_SIZE * IMAGE_SIZE; // 64 pixels
const NUM_CLASSES: usize = 3; // Only 3 classes for simplicity

/// Simple Dense Layer
struct DenseLayer {
    weights: Tensor<CpuBackend, 2>,
    bias: Tensor<CpuBackend, 1>,
}

impl DenseLayer {
    /// Create new dense layer with better initialization
    fn new(input_size: usize, output_size: usize) -> Self {
        // Better initialization: small random values around zero
        let scale = (2.0 / input_size as f32).sqrt();

        let weight_data: Vec<f32> = (0..input_size * output_size)
            .map(|i| {
                let val = ((i * 7 + 13) % 1000) as f32 / 1000.0; // Pseudo-random [0,1]
                (val - 0.5) * scale // Center around 0, scale appropriately
            })
            .collect();

        let bias_data: Vec<f32> = vec![0.0; output_size]; // Initialize bias to zero

        let weights = Tensor::from_data(weight_data, Shape::new([input_size, output_size]));
        let bias = Tensor::from_data(bias_data, Shape::new([output_size]));

        Self { weights, bias }
    }

    /// Forward pass for single input
    fn forward(&self, input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
        // Convert 1D input to 2D for matrix multiplication
        let input_2d = Tensor::from_data(input.to_data(), Shape::new([1, input.shape().dim(0)]));

        // Linear transformation: input @ weights
        let output_2d = input_2d.matmul(&self.weights);

        // Convert back to 1D and add bias
        let linear_output: Tensor<CpuBackend, 1> =
            Tensor::from_data(output_2d.to_data(), Shape::new([self.bias.shape().dim(0)]));

        // Add bias element-wise
        let linear_data = linear_output.to_data();
        let bias_data = self.bias.to_data();
        let result_data: Vec<f32> = linear_data
            .iter()
            .zip(bias_data.iter())
            .map(|(l, b)| l + b)
            .collect();

        Tensor::from_data(result_data, linear_output.shape().clone())
    }
}

/// Simple 3-class classifier
struct SimpleClassifier {
    hidden1: DenseLayer, // 64 -> 32
    hidden2: DenseLayer, // 32 -> 16
    output: DenseLayer,  // 16 -> 3
}

impl SimpleClassifier {
    fn new() -> Self {
        Self {
            hidden1: DenseLayer::new(NUM_PIXELS, 32),
            hidden2: DenseLayer::new(32, 16),
            output: DenseLayer::new(16, NUM_CLASSES),
        }
    }

    /// Forward pass through the network
    fn predict(&self, input: &Tensor<CpuBackend, 1>) -> (usize, f32, Vec<f32>) {
        // Hidden layer 1 + ReLU
        let h1 = self.hidden1.forward(input);
        let h1_activated = h1.relu();

        // Hidden layer 2 + ReLU
        let h2 = self.hidden2.forward(&h1_activated);
        let h2_activated = h2.relu();

        // Output layer + Softmax
        let output = self.output.forward(&h2_activated);
        let probabilities_tensor = output.softmax();
        let probabilities = probabilities_tensor.to_data();

        // Find predicted class
        let (predicted_class, &confidence) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        (predicted_class, confidence, probabilities)
    }
}

/// Generate simple patterns for 3 classes
fn generate_sample_data() -> (Vec<Tensor<CpuBackend, 1>>, Vec<usize>, Vec<String>) {
    let mut images = Vec::new();
    let mut labels = Vec::new();
    let class_names = vec![
        "Circle".to_string(),
        "Cross".to_string(),
        "Square".to_string(),
    ];

    // Generate 5 samples per class
    for class in 0..NUM_CLASSES {
        for variant in 0..5 {
            let mut image_data = vec![0.0; NUM_PIXELS];

            match class {
                0 => generate_circle(&mut image_data, variant),
                1 => generate_cross(&mut image_data, variant),
                2 => generate_square(&mut image_data, variant),
                _ => {}
            }

            images.push(Tensor::from_data(image_data, Shape::new([NUM_PIXELS])));
            labels.push(class);
        }
    }

    (images, labels, class_names)
}

/// Generate circle pattern (Class 0)
fn generate_circle(data: &mut [f32], variant: usize) {
    let center = IMAGE_SIZE as f32 / 2.0;
    let radius = 2.5 + variant as f32 * 0.2; // Slightly different sizes

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let distance = (dx * dx + dy * dy).sqrt();

            if (distance - radius).abs() < 0.8 {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate cross pattern (Class 1)
fn generate_cross(data: &mut [f32], variant: usize) {
    let center = IMAGE_SIZE / 2;
    let thickness = 1 + variant % 2; // Variable thickness

    // Vertical line
    for y in 1..IMAGE_SIZE - 1 {
        for dx in 0..thickness {
            if center + dx < IMAGE_SIZE {
                data[y * IMAGE_SIZE + center + dx] = 1.0;
            }
            if center >= dx {
                data[y * IMAGE_SIZE + center - dx] = 1.0;
            }
        }
    }

    // Horizontal line
    for x in 1..IMAGE_SIZE - 1 {
        for dy in 0..thickness {
            if center + dy < IMAGE_SIZE {
                data[(center + dy) * IMAGE_SIZE + x] = 1.0;
            }
            if center >= dy {
                data[(center - dy) * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate square pattern (Class 2)
fn generate_square(data: &mut [f32], variant: usize) {
    let margin = 1 + variant % 2; // Variable margin

    for y in margin..IMAGE_SIZE - margin {
        for x in margin..IMAGE_SIZE - margin {
            // Draw border only
            if y == margin
                || y == IMAGE_SIZE - margin - 1
                || x == margin
                || x == IMAGE_SIZE - margin - 1
            {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Print 8x8 image in ASCII
fn print_image(image: &Tensor<CpuBackend, 1>, title: &str) {
    println!("{}", title);
    let data = image.to_data();

    for y in 0..IMAGE_SIZE {
        print!("  ");
        for x in 0..IMAGE_SIZE {
            let pixel = data[y * IMAGE_SIZE + x];
            print!("{}", if pixel > 0.5 { "█" } else { "░" });
        }
        println!();
    }
    println!();
}

fn main() {
    println!("=== Simple MNIST Classification with Mini-Burn ===\n");

    // 1. Generate dataset
    println!("1. Creating Dataset");
    let (images, labels, class_names) = generate_sample_data();
    println!(
        "Generated {} images in {} classes",
        images.len(),
        NUM_CLASSES
    );
    println!("Classes: {:?}", class_names);
    println!(
        "Image size: {}x{} = {} pixels\n",
        IMAGE_SIZE, IMAGE_SIZE, NUM_PIXELS
    );

    // 2. Initialize model
    println!("2. Initializing Neural Network");
    let model = SimpleClassifier::new();
    println!(
        "Architecture: {} -> 32 -> 16 -> {}",
        NUM_PIXELS, NUM_CLASSES
    );
    println!("Activation: ReLU (hidden layers) + Softmax (output)\n");

    // 3. Show sample images
    println!("3. Sample Images from Each Class");
    for class in 0..NUM_CLASSES {
        let sample_idx = class * 5; // First sample of each class
        print_image(
            &images[sample_idx],
            &format!("Class {}: {}", class, class_names[class]),
        );
    }

    // 4. Detailed prediction for first few samples
    println!("4. Detailed Predictions");
    for i in 0..6 {
        let (predicted_class, confidence, probabilities) = model.predict(&images[i]);
        let actual_class = labels[i];

        println!("Sample {}: {}", i + 1, class_names[actual_class]);
        print_image(&images[i], "");

        println!("  Predictions:");
        for (class, &prob) in probabilities.iter().enumerate() {
            let marker = if class == predicted_class { " ←" } else { "" };
            println!("    {}: {:.3}{}", class_names[class], prob, marker);
        }

        println!(
            "  Result: {} (confidence: {:.1}%)",
            if predicted_class == actual_class {
                "✓ Correct"
            } else {
                "✗ Wrong"
            },
            confidence * 100.0
        );
        println!();
    }

    // 5. Full evaluation
    println!("5. Full Dataset Evaluation");
    let mut correct = 0;
    let mut class_correct = vec![0; NUM_CLASSES];
    let mut class_total = vec![0; NUM_CLASSES];

    for (i, (image, &actual_label)) in images.iter().zip(labels.iter()).enumerate() {
        let (predicted_label, confidence, _) = model.predict(image);

        class_total[actual_label] += 1;
        if predicted_label == actual_label {
            correct += 1;
            class_correct[actual_label] += 1;
        }

        if i < 3 {
            println!(
                "  Sample {}: Predicted={}, Actual={}, Confidence={:.1}%",
                i + 1,
                class_names[predicted_label],
                class_names[actual_label],
                confidence * 100.0
            );
        }
    }

    println!("\n=== Results ===");
    println!(
        "Overall Accuracy: {:.1}% ({}/{})",
        correct as f32 / images.len() as f32 * 100.0,
        correct,
        images.len()
    );

    println!("\nPer-class Results:");
    for class in 0..NUM_CLASSES {
        let accuracy = if class_total[class] > 0 {
            class_correct[class] as f32 / class_total[class] as f32 * 100.0
        } else {
            0.0
        };
        println!(
            "  {}: {:.1}% ({}/{})",
            class_names[class], accuracy, class_correct[class], class_total[class]
        );
    }

    // 6. Activation analysis
    println!("\n6. Activation Function Demonstration");
    let test_values: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![-1.0, -0.5, 0.0, 0.5, 1.0], Shape::new([5]));

    println!("Test inputs: {:?}", test_values.to_data());
    println!("After ReLU:  {:?}", test_values.relu().to_data());
    println!("After Sigmoid: {:?}", test_values.sigmoid().to_data());

    let softmax_input: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![1.0, 2.0, 3.0], Shape::new([3]));
    println!("Softmax input: {:?}", softmax_input.to_data());
    println!("Softmax output: {:?}", softmax_input.softmax().to_data());

    // 7. Network architecture details
    println!("\n7. Network Architecture Analysis");
    println!("Layer details:");
    println!(
        "  Input: {} neurons ({}x{} flattened image)",
        NUM_PIXELS, IMAGE_SIZE, IMAGE_SIZE
    );
    println!("  Hidden 1: 32 neurons, ReLU activation");
    println!("  Hidden 2: 16 neurons, ReLU activation");
    println!("  Output: {} neurons, Softmax activation", NUM_CLASSES);

    let params = NUM_PIXELS * 32 + 32 + 32 * 16 + 16 + 16 * NUM_CLASSES + NUM_CLASSES;
    println!("Total parameters: {}", params);

    println!("\n=== Demo Complete ===");
    println!("This example demonstrates:");
    println!("✓ Simple image classification");
    println!("✓ Multi-layer neural networks");
    println!("✓ ReLU and Softmax activations");
    println!("✓ Pattern recognition (Circle, Cross, Square)");
    println!("✓ Model evaluation and metrics");
    println!("✓ ASCII visualization");

    println!("\nFramework capabilities shown:");
    println!("✓ Tensor operations (matmul, add, etc.)");
    println!("✓ Activation functions (ReLU, Softmax)");
    println!("✓ Shape management and type safety");
    println!("✓ Forward pass computation");
}
