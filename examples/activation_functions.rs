//! Example demonstrating various activation functions in Mini-Burn
//!
//! This example shows how to use different activation functions commonly
//! used in neural networks and deep learning.

use mini_burn::backend::CpuBackend;
use mini_burn::shape::Shape;
use mini_burn::tensor::Tensor;

fn main() {
    println!("=== Mini-Burn Activation Functions Demo ===\n");

    // Create sample tensors for testing activation functions
    demonstrate_relu();
    demonstrate_sigmoid();
    demonstrate_tanh();
    demonstrate_softmax();
    demonstrate_advanced_activations();
    demonstrate_neural_network_layer();
}

fn demonstrate_relu() {
    println!("1. ReLU (Rectified Linear Unit) Activation");
    println!("   f(x) = max(0, x)");

    let input: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new([5]));

    let output = input.relu();

    println!("   Input:  {:?}", input.to_data());
    println!("   Output: {:?}", output.to_data());
    println!("   Note: All negative values become 0, positive values unchanged\n");
}

fn demonstrate_sigmoid() {
    println!("2. Sigmoid Activation");
    println!("   f(x) = 1 / (1 + exp(-x))");

    let input: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new([5]));

    let output = input.sigmoid();

    println!("   Input:  {:?}", input.to_data());
    println!("   Output: {:?}", output.to_data());
    println!("   Note: Squashes values to range (0, 1)\n");
}

fn demonstrate_tanh() {
    println!("3. Tanh (Hyperbolic Tangent) Activation");
    println!("   f(x) = tanh(x)");

    let input: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new([5]));

    let output = input.tanh();

    println!("   Input:  {:?}", input.to_data());
    println!("   Output: {:?}", output.to_data());
    println!("   Note: Squashes values to range (-1, 1)\n");
}

fn demonstrate_softmax() {
    println!("4. Softmax Activation");
    println!("   Converts logits to probability distribution");

    // 1D Softmax
    println!("   4a. 1D Softmax:");
    let logits_1d: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));

    let probs_1d = logits_1d.softmax();
    let sum_1d: f32 = probs_1d.to_data().iter().sum();

    println!("      Logits: {:?}", logits_1d.to_data());
    println!("      Probs:  {:?}", probs_1d.to_data());
    println!("      Sum:    {:.6} (should be 1.0)", sum_1d);

    // 2D Softmax (batch processing)
    println!("   4b. 2D Softmax (batch of 3 samples, 4 classes each):");
    let logits_2d: Tensor<CpuBackend, 2> = Tensor::from_data(
        vec![
            1.0, 2.0, 3.0, 4.0, // Sample 1
            2.0, 1.0, 4.0, 3.0, // Sample 2
            3.0, 4.0, 1.0, 2.0, // Sample 3
        ],
        Shape::new([3, 4]),
    );

    let probs_2d = logits_2d.softmax();
    let data_2d = probs_2d.to_data();

    println!(
        "      Sample 1 probs: [{:.4}, {:.4}, {:.4}, {:.4}]",
        data_2d[0], data_2d[1], data_2d[2], data_2d[3]
    );
    println!(
        "      Sample 2 probs: [{:.4}, {:.4}, {:.4}, {:.4}]",
        data_2d[4], data_2d[5], data_2d[6], data_2d[7]
    );
    println!(
        "      Sample 3 probs: [{:.4}, {:.4}, {:.4}, {:.4}]",
        data_2d[8], data_2d[9], data_2d[10], data_2d[11]
    );

    // Verify each row sums to 1
    let row1_sum: f32 = data_2d[0..4].iter().sum();
    let row2_sum: f32 = data_2d[4..8].iter().sum();
    let row3_sum: f32 = data_2d[8..12].iter().sum();
    println!(
        "      Row sums: {:.6}, {:.6}, {:.6} (should all be 1.0)\n",
        row1_sum, row2_sum, row3_sum
    );
}

fn demonstrate_advanced_activations() {
    println!("5. Advanced Activation Functions");

    let input: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new([5]));

    // Leaky ReLU
    println!("   5a. Leaky ReLU (alpha=0.1):");
    let leaky_relu_output = input.leaky_relu(0.1);
    println!("      Input:  {:?}", input.to_data());
    println!("      Output: {:?}", leaky_relu_output.to_data());
    println!("      Note: Allows small negative values (0.1 * x for x < 0)");

    // GELU
    println!("   5b. GELU (Gaussian Error Linear Unit):");
    let gelu_output = input.gelu();
    println!("      Input:  {:?}", input.to_data());
    println!("      Output: {:?}", gelu_output.to_data());
    println!("      Note: Smooth approximation to ReLU");

    // Swish/SiLU
    println!("   5c. Swish/SiLU (x * sigmoid(x)):");
    let swish_output = input.swish();
    println!("      Input:  {:?}", input.to_data());
    println!("      Output: {:?}", swish_output.to_data());
    println!("      Note: Self-gated activation function");

    // ELU
    println!("   5d. ELU (Exponential Linear Unit, alpha=1.0):");
    let elu_output = input.elu(1.0);
    println!("      Input:  {:?}", input.to_data());
    println!("      Output: {:?}", elu_output.to_data());
    println!("      Note: Smooth negative part, helps with vanishing gradients\n");
}

fn demonstrate_neural_network_layer() {
    println!("6. Simulated Neural Network Layer");
    println!("   Demonstrating a typical forward pass through a dense layer");

    // Simulate input features (batch_size=2, input_features=3)
    let input: Tensor<CpuBackend, 2> = Tensor::from_data(
        vec![
            0.5, -0.2, 1.0, // Sample 1
            -0.1, 0.8, -0.5, // Sample 2
        ],
        Shape::new([2, 3]),
    );

    // Simulate weights (input_features=3, output_features=4)
    let weights: Tensor<CpuBackend, 2> = Tensor::from_data(
        vec![
            0.2, -0.1, 0.3, 0.4, // Weights for input feature 1
            0.1, 0.5, -0.2, 0.0, // Weights for input feature 2
            -0.3, 0.2, 0.1, 0.6, // Weights for input feature 3
        ],
        Shape::new([3, 4]),
    );

    // Simulate bias (output_features=4)
    let bias: Tensor<CpuBackend, 1> =
        Tensor::from_data(vec![0.1, -0.05, 0.2, 0.0], Shape::new([4]));

    println!("   Input shape: {:?}", input.shape().dims());
    println!("   Weights shape: {:?}", weights.shape().dims());
    println!("   Bias shape: {:?}", bias.shape().dims());

    // Forward pass: output = input @ weights + bias
    let linear_output = input.matmul(&weights);

    // Add bias (broadcasting simulation - we'll add bias to each row)
    let bias_data = bias.to_data();
    let linear_data = linear_output.to_data();
    let mut output_with_bias = vec![0.0; linear_data.len()];

    for i in 0..2 {
        // For each sample
        for j in 0..4 {
            // For each output feature
            output_with_bias[i * 4 + j] = linear_data[i * 4 + j] + bias_data[j];
        }
    }

    let pre_activation: Tensor<CpuBackend, 2> =
        Tensor::from_data(output_with_bias, Shape::new([2, 4]));

    println!("   Linear output (before activation):");
    let pre_data = pre_activation.to_data();
    println!(
        "      Sample 1: [{:.3}, {:.3}, {:.3}, {:.3}]",
        pre_data[0], pre_data[1], pre_data[2], pre_data[3]
    );
    println!(
        "      Sample 2: [{:.3}, {:.3}, {:.3}, {:.3}]",
        pre_data[4], pre_data[5], pre_data[6], pre_data[7]
    );

    // Apply different activation functions
    println!("   After ReLU activation:");
    let relu_output = pre_activation.relu();
    let relu_data = relu_output.to_data();
    println!(
        "      Sample 1: [{:.3}, {:.3}, {:.3}, {:.3}]",
        relu_data[0], relu_data[1], relu_data[2], relu_data[3]
    );
    println!(
        "      Sample 2: [{:.3}, {:.3}, {:.3}, {:.3}]",
        relu_data[4], relu_data[5], relu_data[6], relu_data[7]
    );

    println!("   After Softmax activation (probabilities):");
    let softmax_output = pre_activation.softmax();
    let softmax_data = softmax_output.to_data();
    println!(
        "      Sample 1: [{:.3}, {:.3}, {:.3}, {:.3}]",
        softmax_data[0], softmax_data[1], softmax_data[2], softmax_data[3]
    );
    println!(
        "      Sample 2: [{:.3}, {:.3}, {:.3}, {:.3}]",
        softmax_data[4], softmax_data[5], softmax_data[6], softmax_data[7]
    );

    // Verify softmax probabilities sum to 1
    let sample1_sum: f32 = softmax_data[0..4].iter().sum();
    let sample2_sum: f32 = softmax_data[4..8].iter().sum();
    println!(
        "      Probability sums: {:.6}, {:.6}",
        sample1_sum, sample2_sum
    );

    println!("\n=== Activation Functions Demo Complete ===");
}
