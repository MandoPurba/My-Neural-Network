use crate::{ActivationFunction, DenseLayer, LossFunction, Matrix};

/// Neural Network - menggabungkan multiple layers menjadi satu model
///
/// Network ini bisa:
/// 1. Forward pass - prediksi
/// 2. Backward pass - training via backpropagation
/// 3. Training dengan batch data
/// 4. Evaluasi performa

/// Neural Network struct
/// Menyimpan semua layers dan configurasi
pub struct NeuralNetwork {
    /// Semua layers dalam network, disimpan berurutan
    pub layers: Vec<DenseLayer>,

    /// Activation function untuk setiap layer
    /// Key adalah index layer, Value adalah nama activation
    pub activations: Vec<String>,

    /// Statistik training untuk monitoring
    pub training_stats: TrainingStats,
}

/// Statistik training untuk monitoring proses pembelajaran
#[derive(Debug, Clone)]
pub struct TrainingStats {
    pub epochs_trained: usize,
    pub loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
}

impl TrainingStats {
    pub fn new() -> Self {
        Self {
            epochs_trained: 0,
            loss_history: Vec::new(),
            accuracy_history: Vec::new(),
        }
    }
}

impl NeuralNetwork {
    /// Membuat Neural Network baru yang kosong
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            activations: Vec::new(),
            training_stats: TrainingStats::new(),
        }
    }

    /// Menambahkan Dense Layer ke network
    ///
    /// # Arguments
    /// * `input_size` - Ukuran input (untuk layer pertama) atau otomatis dari layer sebelumnya
    /// * `output_size` - Ukuran output layer ini
    /// * `activation` - Nama activation function: "relu", "sigmoid", "tanh", "linear"
    ///
    /// # Example
    /// ```
    /// let mut network = NeuralNetwork::new();
    /// network.add_dense_layer(784, 128, "relu");  // Input layer
    /// network.add_dense_layer(128, 64, "relu");   // Hidden layer
    /// network.add_dense_layer(64, 10, "sigmoid"); // Output layer
    /// ```
    pub fn add_dense_layer(&mut self, input_size: usize, output_size: usize, activation: &str) {
        // Validasi ukuran input untuk layer kedua dan seterusnya
        if !self.layers.is_empty() {
            let last_layer = &self.layers[self.layers.len() - 1];
            assert_eq!(
                input_size, last_layer.output_size,
                "Input size {} must match previous layer output size {}",
                input_size, last_layer.output_size
            );
        }

        // FIX: Pindahkan pembuatan layer ke luar if statement
        let layer = DenseLayer::new(input_size, output_size);
        self.layers.push(layer);
        self.activations.push(activation.to_string());
    }

    /// Helper method untuk membuat network dengan architecture yang mudah
    ///
    /// # Arguments
    /// * `layer_sizes` - Array ukuran setiap layer [input_size, hidden1, hidden2, ..., output_size]
    /// * `hidden_activation` - Activation untuk hidden layers
    /// * `output_activation` - Activation untuk output layer
    ///
    /// # Example
    /// ```
    /// // Network untuk MNIST: 784 -> 128 -> 64 -> 10
    /// let mut network = NeuralNetwork::from_architecture(
    ///    &[784, 128, 64, 10],
    ///    "relu",
    ///    "sigmoid"
    /// );
    /// ```
    pub fn from_architecture(
        layer_sizes: &[usize],
        hidden_activation: &str,
        output_activation: &str,
    ) -> Self {
        assert!(
            layer_sizes.len() >= 2,
            "Need at least input and output layers"
        );

        let mut network = NeuralNetwork::new();

        // Tambahkan semua layers
        for i in 0..layer_sizes.len() - 1 {
            let activation = if i == layer_sizes.len() - 2 {
                output_activation // Last layer
            } else {
                hidden_activation // Hidden layers
            };

            network.add_dense_layer(layer_sizes[i], layer_sizes[i + 1], activation);
        }

        network
    }

    /// Forward pass - prediksi
    /// Melewatkan input melalui semua layers
    ///
    /// # Arguments
    /// * `input` - Input matrix [batch_size x input_features]
    ///
    /// # Returns
    /// Output predictions [batch_size x output_features]
    pub fn predict(&mut self, input: &Matrix) -> Matrix {
        let mut current_input = input.clone();

        // Debug: Print dimensions di setiap layer
        println!("Input shape: {}x{}", current_input.rows, current_input.cols);

        // Lewati setiap layer satu per satu
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let activation = get_activation_function(&self.activations[i]);
            current_input = layer.forward(&current_input, activation.as_ref());

            // Debug: Print output shape setiap layer
            println!(
                "Layer {} output shape: {}x{}",
                i + 1,
                current_input.rows,
                current_input.cols
            );
        }

        current_input
    }

    /// Training satu batch data
    ///
    /// # Arguments
    /// * `inputs` - Training inputs [batch_size x input_features]
    /// * `targets` - True labels [batch_size x output_features]
    /// * `loss_function` - Loss function untuk menghitung error
    /// * `learning_rate` - Learning rate untuk gradient descent
    ///
    /// # Returns
    /// Loss value untuk batch ini
    pub fn train_batch(
        &mut self,
        inputs: &Matrix,
        targets: &Matrix,
        loss_function: &dyn LossFunction,
        learning_rate: f32,
    ) -> f32 {
        // Forward pass
        let predictions = self.predict(inputs);

        // Hitung loss
        let loss = loss_function.calculate_loss(&predictions, targets);

        // Backward pass (backpropagation)
        let mut current_gradient = loss_function.calculate_gradient(&predictions, targets);

        // Backpropagate melalui setiap layer (dari belakang ke depan)
        for i in (0..self.layers.len()).rev() {
            let activation = get_activation_function(&self.activations[i]);
            current_gradient =
                self.layers[i].backward(&current_gradient, activation.as_ref(), learning_rate);
        }
        loss
    }

    /// Training untuk multiple epochs
    ///
    /// # Arguments
    /// * `train_inputs` - Training data inputs
    /// * `train_targets` - Training data targets
    /// * `loss_function` - Loss function
    /// * `learning_rate` - Learning rate
    /// * `epochs` - Jumlah epochs
    /// * `verbose` - Print progress atau tidak
    pub fn train(
        &mut self,
        train_inputs: &Matrix,
        train_targets: &Matrix,
        loss_function: &dyn LossFunction,
        learning_rate: f32,
        epochs: usize,
        verbose: bool,
    ) {
        for epoch in 0..epochs {
            let loss = self.train_batch(train_inputs, train_targets, loss_function, learning_rate);

            // Simpan statistik
            self.training_stats.loss_history.push(loss);
            self.training_stats.epochs_trained += 1;

            // Print progress
            if verbose && (epoch % 100 == 0 || epoch == epochs - 1) {
                println!("Epoch {}/{}, Loss: {:.6}", epoch + 1, epochs, loss);
            }
        }
    }

    /// Hitung accuracy untuk classification task
    pub fn calculate_accuracy(&mut self, inputs: &Matrix, targets: &Matrix) -> f32 {
        let predictions = self.predict(inputs);
        let mut correct = 0;
        let total = predictions.rows;

        for i in 0..total {
            // Untuk binary classification
            if predictions.cols == 1 {
                let pred = if predictions.get(i, 0) > 0.5 {
                    1.0
                } else {
                    0.0
                };
                // FIX: Typo - seharusnya targets.get(i, 0) bukan targets.get(1, 0)
                if (pred - targets.get(i, 0)).abs() < 0.1 {
                    correct += 1;
                }
            } else {
                // Untuk multi-class classification - ambil index dengan nilai tertinggi
                let mut max_pred_idx = 0;
                let mut max_target_idx = 0;

                for j in 1..predictions.cols {
                    if predictions.get(i, j) > predictions.get(i, max_pred_idx) {
                        max_pred_idx = j;
                    }
                    if targets.get(i, j) > targets.get(i, max_target_idx) {
                        max_target_idx = j;
                    }
                }

                if max_pred_idx == max_target_idx {
                    correct += 1;
                }
            }
        }
        correct as f32 / total as f32
    }

    /// Print summary network architecture
    pub fn summary(&self) {
        println!("Neural Network Summary:");
        println!("======================");

        let mut total_params = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            let params = layer.parameter_count();
            total_params += params;

            println!(
                "Layer {}: {} -> {} ({} activation) | Parameters: {}",
                i + 1,
                layer.input_size,
                layer.output_size,
                self.activations[i],
                params
            );
        }

        println!("======================");
        println!("Total Parameters: {}", total_params);
        println!("Epochs Trained: {}", self.training_stats.epochs_trained);
    }
}

/// Helper function untuk mendapatkan activation function dari string
fn get_activation_function(name: &str) -> Box<dyn ActivationFunction> {
    match name.to_lowercase().as_str() {
        "relu" => Box::new(crate::activation::ReLU),
        "sigmoid" => Box::new(crate::activation::Sigmoid),
        "tanh" => Box::new(crate::activation::Tanh),
        "linear" => Box::new(crate::activation::Linear),
        _ => panic!("Unknown activation function: {}", name),
    }
}
