use mini_burn::{BinaryCrossEntropy, Matrix, NeuralNetwork};

fn main() {
    println!("XOR Problem (Classification)");
    println!("============================");

    // XOR training data
    // Input: [x1, x2], Output: [x1 XOR x2]
    let train_inputs = Matrix::from_data(
        vec![
            0.0, 0.0, // 0 XOR 0 = 0
            0.0, 0.1, // 0 XOR 1 = 1
            1.0, 0.0, // 1 XOR 0 = 1
            1.0, 1.0, // 1 XOR 1 = 0
        ],
        4,
        2,
    );

    let train_targets = Matrix::from_data(vec![0.0, 1.0, 1.0, 0.0], 4, 1);

    // Buat network: 2 input -> hidden -> 1 output
    let mut network = NeuralNetwork::from_architecture(&[2, 4, 1], "relu", "sigmoid");

    network.summary();

    // Training
    let loss_fn = BinaryCrossEntropy;
    println!("\n🎯 Training on XOR data...");
    network.train(&train_inputs, &train_targets, &loss_fn, 0.1, 1000, true);

    // Testing
    println!("\n🧪 Testing XOR predictions:");
    let predictions = network.predict(&train_inputs);

    for i in 0..4 {
        let x1 = train_inputs.get(i, 0);
        let x2 = train_inputs.get(i, 1);
        let expected = train_targets.get(i, 0);
        let predicted = predictions.get(i, 0);

        println!(
            "{:.0} XOR {:.0} = {:.0} | Predicted: {:.4} | Correct: {}",
            x1,
            x2,
            expected,
            predicted,
            if (predicted - expected).abs() < 0.3 {
                "✅"
            } else {
                "❌"
            }
        );
    }

    let accuracy = network.calculate_accuracy(&train_inputs, &train_targets);
    println!("Accuracy: {:.2}%\n", accuracy * 100.0);
}
