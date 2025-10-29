//! MNIST with High-Level Neural Network APIs
//!
//! This example demonstrates how to use Mini-Burn's high-level neural network APIs
//! to build and evaluate a model for MNIST-like digit classification.
//! This version is much simpler and cleaner than the manual implementation.

use mini_burn::backend::CpuBackend;
use mini_burn::nn::{
    layers::{Linear, ReLU, Softmax},
    loss::{CrossEntropyLoss, Loss, LossUtils},
    ModelBuilder, Sequential, Trainer,
};
use mini_burn::shape::Shape;
use mini_burn::tensor::Tensor;

const IMAGE_SIZE: usize = 16; // Simplified 16x16 for faster demo
const NUM_PIXELS: usize = IMAGE_SIZE * IMAGE_SIZE; // 256 pixels
const NUM_CLASSES: usize = 10; // Digits 0-9

fn main() {
    println!("=== MNIST with High-Level Neural Network APIs ===\n");

    // 1. Create model using high-level APIs
    println!("1. Building Neural Network Model");

    // Option 1: Using ModelBuilder (automatic architecture)
    let classifier = ModelBuilder::<CpuBackend>::classifier(
        NUM_PIXELS,  // 256 input features
        &[128, 64],  // hidden layers
        NUM_CLASSES, // 10 output classes
    );

    println!("Created classifier using ModelBuilder:");
    println!(
        "  Architecture: {} → 128 → 64 → {}",
        NUM_PIXELS, NUM_CLASSES
    );
    println!("  Total layers: {}", classifier.len());

    // Option 2: Using Sequential API (custom architecture)
    let custom_model = Sequential::<CpuBackend>::new()
        .add(Linear::new(NUM_PIXELS, 200))
        .add(ReLU::new())
        .add(Linear::new(200, 100))
        .add(ReLU::new())
        .add(Linear::new(100, 50))
        .add(ReLU::new())
        .add(Linear::new(50, NUM_CLASSES))
        .add(Softmax::new());

    println!("Created custom model using Sequential API:");
    println!(
        "  Architecture: {} → 200 → 100 → 50 → {}",
        NUM_PIXELS, NUM_CLASSES
    );
    println!("  Total layers: {}\n", custom_model.len());

    // 2. Generate MNIST-like dataset
    println!("2. Generating MNIST-like Dataset");
    let (images, labels) = generate_mnist_dataset();
    println!(
        "Generated {} images for {} digits",
        images.len(),
        NUM_CLASSES
    );
    println!(
        "Image dimensions: {}x{} = {} pixels\n",
        IMAGE_SIZE, IMAGE_SIZE, NUM_PIXELS
    );

    // 3. Display sample images
    println!("3. Sample Images Visualization");
    for digit in 0..5 {
        println!("Digit {} sample:", digit);
        print_image(&images[digit * 2]); // Show first sample of each digit
        println!();
    }

    // 4. Model evaluation using high-level APIs
    println!("4. Model Evaluation");

    // Evaluate ModelBuilder classifier
    println!("Evaluating ModelBuilder classifier:");
    let (accuracy1, class_accs1) = Trainer::evaluate(&classifier, &images, &labels);
    println!("  Overall Accuracy: {:.1}%", accuracy1 * 100.0);
    println!("  Per-class Accuracy:");
    for (digit, &acc) in class_accs1.iter().enumerate() {
        println!("    Digit {}: {:.1}%", digit, acc * 100.0);
    }

    // Evaluate custom model
    println!("\nEvaluating custom Sequential model:");
    let (accuracy2, class_accs2) = Trainer::evaluate(&custom_model, &images, &labels);
    println!("  Overall Accuracy: {:.1}%", accuracy2 * 100.0);
    println!("  Per-class Accuracy:");
    for (digit, &acc) in class_accs2.iter().enumerate() {
        println!("    Digit {}: {:.1}%", digit, acc * 100.0);
    }

    // 5. Loss computation
    println!("\n5. Loss Function Evaluation");
    let ce_loss = CrossEntropyLoss::<CpuBackend>::new();
    let mut total_loss = 0.0;
    let num_samples = images.len().min(10); // Evaluate first 10 samples

    for i in 0..num_samples {
        let prediction = classifier.predict(&images[i]);
        let pred_batch = Tensor::from_data(prediction.to_data(), Shape::new([1, NUM_CLASSES]));
        let loss = ce_loss.compute_with_indices(&pred_batch, &[labels[i]]);
        total_loss += loss;

        if i < 3 {
            println!("  Sample {}: Loss = {:.4}", i + 1, loss);
        }
    }

    let avg_loss = total_loss / num_samples as f32;
    println!("  Average Cross Entropy Loss: {:.4}", avg_loss);

    // 6. Detailed predictions
    println!("\n6. Detailed Predictions");
    for i in 0..6 {
        let prediction = classifier.predict(&images[i]);
        let probs = prediction.to_data();

        let (predicted_digit, &confidence) = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        let actual_digit = labels[i];
        let status = if predicted_digit == actual_digit {
            "✓"
        } else {
            "✗"
        };

        println!(
            "Sample {}: Actual={}, Predicted={}, Confidence={:.1}% {}",
            i + 1,
            actual_digit,
            predicted_digit,
            confidence * 100.0,
            status
        );

        // Show top 3 predictions
        let mut indexed_probs: Vec<(usize, f32)> =
            probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        println!("  Top 3: ");
        for (rank, (digit, prob)) in indexed_probs.iter().take(3).enumerate() {
            println!("    {}: Digit {} ({:.1}%)", rank + 1, digit, prob * 100.0);
        }
    }

    // 7. Batch processing demonstration
    println!("\n7. Batch Processing");
    let batch_size = 4;
    let mut batch_data = Vec::new();

    // Create batch from first 4 images
    for i in 0..batch_size {
        batch_data.extend(images[i].to_data());
    }

    let batch = Tensor::from_data(batch_data, Shape::new([batch_size, NUM_PIXELS]));
    let batch_predictions = Trainer::batch_predict(&classifier, &batch);

    println!("Batch processing {} images:", batch_size);
    for (i, (predicted_class, confidence)) in batch_predictions.iter().enumerate() {
        println!(
            "  Image {}: Predicted={}, Actual={}, Confidence={:.1}%",
            i + 1,
            predicted_class,
            labels[i],
            confidence * 100.0
        );
    }

    // 8. Model comparison
    println!("\n8. Model Comparison");
    println!("ModelBuilder Classifier vs Custom Sequential Model:");
    println!("  ModelBuilder Accuracy: {:.1}%", accuracy1 * 100.0);
    println!("  Custom Model Accuracy: {:.1}%", accuracy2 * 100.0);
    println!(
        "  Difference: {:.1}%",
        (accuracy2 - accuracy1).abs() * 100.0
    );

    println!("\n=== Demo Complete ===");
    println!("This example demonstrated:");
    println!("✓ High-level model building (ModelBuilder & Sequential)");
    println!("✓ Automatic model evaluation with Trainer");
    println!("✓ Loss function computation");
    println!("✓ Batch processing capabilities");
    println!("✓ Easy model comparison");
    println!("✓ Clean, readable code using high-level APIs");
    println!("✓ No manual forward pass implementation needed!");
}

/// Generate synthetic MNIST-like dataset
fn generate_mnist_dataset() -> (Vec<Tensor<CpuBackend, 1>>, Vec<usize>) {
    let mut images = Vec::new();
    let mut labels = Vec::new();

    // Generate 3 samples per digit
    for digit in 0..NUM_CLASSES {
        for sample in 0..3 {
            let mut image_data = vec![0.0; NUM_PIXELS];

            // Generate digit patterns
            match digit {
                0 => generate_circle(&mut image_data, sample),
                1 => generate_vertical_line(&mut image_data, sample),
                2 => generate_horizontal_lines(&mut image_data, sample),
                3 => generate_curves(&mut image_data, sample),
                4 => generate_triangle(&mut image_data, sample),
                5 => generate_square(&mut image_data, sample),
                6 => generate_oval(&mut image_data, sample),
                7 => generate_diagonal(&mut image_data, sample),
                8 => generate_cross(&mut image_data, sample),
                9 => generate_spiral(&mut image_data, sample),
                _ => {}
            }

            images.push(Tensor::from_data(image_data, Shape::new([NUM_PIXELS])));
            labels.push(digit);
        }
    }

    (images, labels)
}

/// Generate circle pattern for digit 0
fn generate_circle(data: &mut [f32], variant: usize) {
    let center = IMAGE_SIZE as f32 / 2.0;
    let radius = 4.0 + variant as f32 * 0.5;

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let dx = x as f32 - center;
            let dy = y as f32 - center;
            let distance = (dx * dx + dy * dy).sqrt();

            if (distance - radius).abs() < 1.0 {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate vertical line for digit 1
fn generate_vertical_line(data: &mut [f32], variant: usize) {
    let center_x = IMAGE_SIZE / 2 + variant % 2;
    let thickness = 1 + variant % 2;

    for y in 2..IMAGE_SIZE - 2 {
        for dx in 0..thickness {
            if center_x + dx < IMAGE_SIZE {
                data[y * IMAGE_SIZE + center_x + dx] = 1.0;
            }
        }
    }
}

/// Generate horizontal lines for digit 2
fn generate_horizontal_lines(data: &mut [f32], variant: usize) {
    let lines = [3 + variant, 8, 13 - variant];

    for &line_y in &lines {
        if line_y < IMAGE_SIZE {
            for x in 2..IMAGE_SIZE - 2 {
                data[line_y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate curves for digit 3
fn generate_curves(data: &mut [f32], variant: usize) {
    let offset = variant as f32 * 0.5;

    for y in 0..IMAGE_SIZE {
        let x1 = (6.0 + 3.0 * (y as f32 * 0.4 + offset).sin()) as usize;
        let x2 = (10.0 + 3.0 * (y as f32 * 0.4 + offset).cos()) as usize;

        if x1 < IMAGE_SIZE {
            data[y * IMAGE_SIZE + x1] = 1.0;
        }
        if x2 < IMAGE_SIZE {
            data[y * IMAGE_SIZE + x2] = 1.0;
        }
    }
}

/// Generate triangle for digit 4
fn generate_triangle(data: &mut [f32], variant: usize) {
    let offset = variant;

    for y in 3..IMAGE_SIZE - 3 {
        for x in 3..IMAGE_SIZE - 3 {
            if x == 6 + offset || y == 8 || (x + y == 14 + offset && x < 10) {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate square for digit 5
fn generate_square(data: &mut [f32], variant: usize) {
    let margin = 3 + variant % 2;

    for y in margin..IMAGE_SIZE - margin {
        for x in margin..IMAGE_SIZE - margin {
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

/// Generate oval for digit 6
fn generate_oval(data: &mut [f32], variant: usize) {
    let center_x = 8.0;
    let center_y = 8.0 + variant as f32 * 0.5;

    for y in 0..IMAGE_SIZE {
        for x in 0..IMAGE_SIZE {
            let dx = (x as f32 - center_x) / 3.0;
            let dy = (y as f32 - center_y) / 4.0;

            if (dx * dx + dy * dy - 1.0).abs() < 0.4 {
                data[y * IMAGE_SIZE + x] = 1.0;
            }
        }
    }
}

/// Generate diagonal for digit 7
fn generate_diagonal(data: &mut [f32], variant: usize) {
    let start_offset = variant;

    for y in 2..IMAGE_SIZE - 2 {
        let x = (2 + start_offset + y * 2 / 3) % (IMAGE_SIZE - 4) + 2;
        if x < IMAGE_SIZE {
            data[y * IMAGE_SIZE + x] = 1.0;
            if x > 0 {
                data[y * IMAGE_SIZE + x - 1] = 0.5;
            }
        }
    }
}

/// Generate cross for digit 8
fn generate_cross(data: &mut [f32], variant: usize) {
    let center = IMAGE_SIZE / 2;
    let offset = variant % 2;

    // Vertical line
    for y in 3..IMAGE_SIZE - 3 {
        data[y * IMAGE_SIZE + center + offset] = 1.0;
    }

    // Horizontal line
    for x in 3..IMAGE_SIZE - 3 {
        data[(center + offset) * IMAGE_SIZE + x] = 1.0;
    }
}

/// Generate spiral for digit 9
fn generate_spiral(data: &mut [f32], variant: usize) {
    let center_x = 8.0;
    let center_y = 8.0;
    let spiral_tightness = 1.0 + variant as f32 * 0.1;

    for i in 0..50 {
        let angle = i as f32 * 0.4 * spiral_tightness;
        let radius = i as f32 * 0.08;

        let x = (center_x + radius * angle.cos()) as usize;
        let y = (center_y + radius * angle.sin()) as usize;

        if x < IMAGE_SIZE && y < IMAGE_SIZE {
            data[y * IMAGE_SIZE + x] = 1.0;
        }
    }
}

/// Print image in ASCII format
fn print_image(image: &Tensor<CpuBackend, 1>) {
    let data = image.to_data();

    for y in 0..IMAGE_SIZE {
        print!("  ");
        for x in 0..IMAGE_SIZE {
            let pixel = data[y * IMAGE_SIZE + x];
            let char = if pixel > 0.7 {
                '█'
            } else if pixel > 0.3 {
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
