//! Simple neural network implementation using mini-burn framework
//! This demonstrates how our tensor operations can be used to build basic neural networks

use mini_burn::{CpuBackend, Shape, Tensor};

/// Simple linear layer implementation
struct LinearLayer {
    weights: Tensor<CpuBackend, 2>,
    bias: Tensor<CpuBackend, 1>,
}

impl LinearLayer {
    /// Create a new linear layer with random-ish initialization
    fn new(input_size: usize, output_size: usize) -> Self {
        // Simple initialization - in real networks you'd use proper initialization
        let weight_data: Vec<f32> = (0..input_size * output_size)
            .map(|i| (i as f32 * 0.1) % 1.0 - 0.5) // Simple pseudo-random values
            .collect();

        let bias_data: Vec<f32> = (0..output_size).map(|i| (i as f32 * 0.05) % 0.5).collect();

        let weights = Tensor::from_data(weight_data, Shape::new([input_size, output_size]));
        let bias = Tensor::from_data(bias_data, Shape::new([output_size]));

        Self { weights, bias }
    }

    /// Forward pass: input @ weights + bias
    /// Note: This is a simplified version, real implementation would handle batching
    fn forward(&self, input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
        // Convert 1D input to 2D for matrix multiplication
        let input_2d: Tensor<CpuBackend, 2> =
            Tensor::from_data(input.to_data(), Shape::new([1, input.shape().dim(0)]));

        // Matrix multiplication: [1, input_size] @ [input_size, output_size] = [1, output_size]
        let output_2d = input_2d.matmul(&self.weights);

        // Convert back to 1D and add bias
        let output_1d: Tensor<CpuBackend, 1> =
            Tensor::from_data(output_2d.to_data(), Shape::new([self.bias.shape().dim(0)]));

        // Add bias (element-wise addition)
        let output_data: Vec<f32> = output_1d
            .to_data()
            .iter()
            .zip(self.bias.to_data().iter())
            .map(|(o, b)| o + b)
            .collect();

        Tensor::from_data(output_data, output_1d.shape().clone())
    }
}

/// Simple activation functions
fn relu(input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
    let data: Vec<f32> = input
        .to_data()
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect();

    Tensor::from_data(data, input.shape().clone())
}

fn sigmoid(input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
    let data: Vec<f32> = input
        .to_data()
        .iter()
        .map(|&x| 1.0 / (1.0 + (-x).exp()))
        .collect();

    Tensor::from_data(data, input.shape().clone())
}

fn tanh_activation(input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
    let data: Vec<f32> = input.to_data().iter().map(|&x| x.tanh()).collect();

    Tensor::from_data(data, input.shape().clone())
}

/// Simple 2-layer neural network
struct SimpleNetwork {
    layer1: LinearLayer,
    layer2: LinearLayer,
}

impl SimpleNetwork {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> Self {
        Self {
            layer1: LinearLayer::new(input_size, hidden_size),
            layer2: LinearLayer::new(hidden_size, output_size),
        }
    }

    fn forward(&self, input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
        // Layer 1 + ReLU activation
        let hidden = self.layer1.forward(input);
        let hidden_activated = relu(&hidden);

        // Layer 2 + Sigmoid activation
        let output = self.layer2.forward(&hidden_activated);
        sigmoid(&output)
    }
}

/// Multi-layer neural network with configurable architecture
struct MultiLayerNetwork {
    layers: Vec<LinearLayer>,
}

impl MultiLayerNetwork {
    fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            layers.push(LinearLayer::new(layer_sizes[i], layer_sizes[i + 1]));
        }

        Self { layers }
    }

    fn forward(&self, input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
        let mut current = input.clone();

        for (i, layer) in self.layers.iter().enumerate() {
            current = layer.forward(&current);

            // Apply activation function (except for the last layer)
            if i < self.layers.len() - 1 {
                current = tanh_activation(&current);
            } else {
                // Apply sigmoid to output layer
                current = sigmoid(&current);
            }
        }

        current
    }
}

fn main() {
    println!("=== Simple Neural Network with Mini-Burn ===\n");

    // 1. Create a simple network
    println!("1. Creating Neural Network:");
    let network = SimpleNetwork::new(4, 6, 2); // 4 inputs, 6 hidden, 2 outputs
    println!("Created network: 4 -> 6 -> 2");

    // 2. Create some sample input
    println!("\n2. Sample Input:");
    let input = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));
    println!("Input: {:?}", input.to_data());

    // 3. Forward pass
    println!("\n3. Forward Pass:");
    let output = network.forward(&input);
    println!("Output: {:?}", output.to_data());
    println!("Output shape: {:?}", output.shape().dims());

    // 4. Test with different inputs
    println!("\n4. Multiple Inputs:");
    let inputs = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
        vec![1.0, 1.0, 1.0, 1.0],
    ];

    for (i, input_data) in inputs.iter().enumerate() {
        let input_tensor = Tensor::from_data(input_data.clone(), Shape::new([4]));
        let output = network.forward(&input_tensor);
        println!(
            "Input {}: {:?} -> Output: {:?}",
            i + 1,
            input_data,
            output.to_data()
        );
    }

    // 5. Demonstrate layer components
    println!("\n5. Layer Components:");
    println!(
        "Layer 1 weights shape: {:?}",
        network.layer1.weights.shape().dims()
    );
    println!(
        "Layer 1 bias shape: {:?}",
        network.layer1.bias.shape().dims()
    );
    println!(
        "Layer 2 weights shape: {:?}",
        network.layer2.weights.shape().dims()
    );
    println!(
        "Layer 2 bias shape: {:?}",
        network.layer2.bias.shape().dims()
    );

    // 6. Manual activation function testing
    println!("\n6. Activation Functions:");
    let test_values = Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new([5]));
    println!("Test values: {:?}", test_values.to_data());

    let relu_result = relu(&test_values);
    println!("ReLU result: {:?}", relu_result.to_data());

    let sigmoid_result = sigmoid(&test_values);
    println!("Sigmoid result: {:?}", sigmoid_result.to_data());

    let tanh_result = tanh_activation(&test_values);
    println!("Tanh result: {:?}", tanh_result.to_data());

    // 7. Multi-layer network demonstration
    println!("\n7. Multi-Layer Network:");
    let deep_network = MultiLayerNetwork::new(&[3, 8, 6, 4, 2]); // 5-layer network
    println!("Created deep network: 3 -> 8 -> 6 -> 4 -> 2");

    let deep_input = Tensor::from_data(vec![0.5, -0.3, 1.2], Shape::new([3]));
    println!("Deep network input: {:?}", deep_input.to_data());

    let deep_output = deep_network.forward(&deep_input);
    println!("Deep network output: {:?}", deep_output.to_data());

    // 8. Simulating a classification task
    println!("\n8. Simulated Classification Task:");
    let classifier = SimpleNetwork::new(2, 4, 3); // 2D input, 3 classes

    let test_points = vec![
        vec![0.1, 0.9],  // Point 1
        vec![0.8, 0.2],  // Point 2
        vec![0.5, 0.5],  // Point 3
        vec![-0.3, 0.7], // Point 4
        vec![0.9, -0.1], // Point 5
    ];

    println!("Classification results:");
    for (i, point) in test_points.iter().enumerate() {
        let input_tensor = Tensor::from_data(point.clone(), Shape::new([2]));
        let probabilities = classifier.forward(&input_tensor);
        let probs = probabilities.to_data();

        // Find the class with highest probability
        let predicted_class = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(index, _)| index)
            .unwrap();

        println!(
            "Point {}: {:?} -> Probabilities: {:?}, Predicted class: {}",
            i + 1,
            point,
            probs,
            predicted_class
        );
    }

    // 9. Demonstrating network capacity
    println!("\n9. Network Capacity Demonstration:");
    let tiny_net = SimpleNetwork::new(1, 2, 1);
    let medium_net = MultiLayerNetwork::new(&[1, 5, 5, 1]);
    let large_net = MultiLayerNetwork::new(&[1, 10, 8, 6, 1]);

    let x_values = vec![-2.0, -1.0, 0.0, 1.0, 2.0];

    println!("Input values: {:?}", x_values);

    for net_name in &["Tiny", "Medium", "Large"] {
        print!("{} network outputs: [", net_name);
        for (i, &x) in x_values.iter().enumerate() {
            let input_tensor = Tensor::from_data(vec![x], Shape::new([1]));
            let output = match *net_name {
                "Tiny" => tiny_net.forward(&input_tensor),
                "Medium" => medium_net.forward(&input_tensor),
                "Large" => large_net.forward(&input_tensor),
                _ => unreachable!(),
            };

            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", output.to_data()[0]);
        }
        println!("]");
    }

    println!("\n=== Neural Network Demo Complete ===");
    println!("This demonstrates basic building blocks for deep learning:");
    println!("- Linear layers with matrix multiplication");
    println!("- Activation functions (ReLU, Sigmoid, Tanh)");
    println!("- Forward pass through multiple layers");
    println!("- Multi-layer architectures");
    println!("- Classification simulation");
    println!("- Type-safe tensor operations");
}
