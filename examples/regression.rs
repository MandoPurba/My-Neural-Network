use mini_burn::{Matrix, MeanSquaredError, NeuralNetwork};

fn main() {
    println!("📈 Demo 2: Simple Regression y = 2x + 1");
    println!("----------------------------------------");

    // Regression training data: y = 2x + 1
    let train_inputs = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);

    let train_targets = Matrix::from_data(
        vec![3.0, 5.0, 7.0, 9.0, 11.0], // 2*x + 1
        5,
        1,
    );

    // Buat simple network: 1 -> 3 -> 1
    let mut network = NeuralNetwork::from_architecture(
        &[1, 3, 1],
        "relu",
        "linear", // Linear output untuk regression
    );

    network.summary();

    // Training
    let loss_fn = MeanSquaredError;
    println!("\n🎯 Training regression model...");
    network.train(&train_inputs, &train_targets, &loss_fn, 0.01, 1000, true);

    // Testing
    println!("\n🧪 Testing predictions:");
    let predictions = network.predict(&train_inputs);

    for i in 0..5 {
        let x = train_inputs.get(i, 0);
        let expected = train_targets.get(i, 0);
        let predicted = predictions.get(i, 0);
        let error = (predicted - expected).abs();

        println!(
            "x={:.0} | Expected: {:.1} | Predicted: {:.3} | Error: {:.3}",
            x, expected, predicted, error
        );
    }

    println!();
}
