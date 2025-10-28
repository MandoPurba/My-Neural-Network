/// Layer adalah komponen building block dari Neural Network
/// Dense Layer (Fully Connected Layer) adalah yang paling umum
///
/// Setiap Dense Layer melakukan operasi: output = activation(input * weights + bias)
///
/// Contoh: Jika input 3 features dan output 2 neurons:
/// - weights matrix: 3x2
/// - bias vector: 1x2
/// - input: 1x3 atau batch_size x 3
/// - output: 1x2 atau batch_size x 2
use crate::{ActivationFunction, Matrix};

/// Dense Layer - Fully Connected Layer
/// Ini adalah lapisan paling dasar dalam Neural Network
#[derive(Debug, Clone)]
pub struct DenseLayer {
    /// Weights matrix: [input_size x output_size]
    /// Setiap kolom adalah weights untuk satu neuron output
    pub weights: Matrix,

    /// Bias vector: [1 x output_size]
    /// Setiap neuron punya bias sendiri
    pub bias: Matrix,

    /// Ukuran input dan output
    pub input_size: usize,
    pub output_size: usize,

    /// Menyimpan input terakhir untuk backpropagation
    /// Kita perlu ini untuk menghitung gradient
    pub last_input: Option<Matrix>,

    /// Menyimpan output sebelum activation untuk backpropagation
    pub last_output: Option<Matrix>,
}

impl DenseLayer {
    /// Membuat Dense Layer baru
    ///
    /// # Arguments
    /// * `input_size` - Jumlah input features
    /// * `output_size` - Jumlah output neurons
    ///
    /// # Example
    /// ```
    /// // Layer dengan 3 input dan 5 output neurons
    /// let layer = DenseLayer::new(3, 5);
    /// ```
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            // Inisialisasi weights dengan Xavier initialization
            weights: Matrix::random(input_size, output_size),
            // Bias dimulai dari nol
            bias: Matrix::new(1, output_size),
            input_size,
            output_size,
            last_input: None,
            last_output: None,
        }
    }

    /// Forward pass tanpa activation function
    /// Melakukan: output = input * weights + bias
    ///
    /// # Arguments
    /// * `input` - Input matrix [batch_size x input_size] atau [1 x input_size]
    ///
    /// # Returns
    /// Output matrix [batch_size x output_size]
    pub fn forward_raw(&mut self, input: &Matrix) -> Matrix {
        assert_eq!(
            input.cols, self.input_size,
            "Input columns ({}) must match layer input size ({})",
            input.cols, self.input_size
        );

        // Simpan input untuk backpropagation
        self.last_input = Some(input.clone());

        // Matrix multiplication: input * weights
        let weighted = input.multiply(&self.weights);

        // Add bias ke setiap row (setiap sample dalam batch)
        let mut result = Matrix::new(weighted.rows, weighted.cols);
        for i in 0..weighted.rows {
            for j in 0..weighted.cols {
                result.set(i, j, weighted.get(i, j) + self.bias.get(0, j));
            }
        }

        // Simpan output sebelum activation
        self.last_output = Some(result.clone());
        result
    }

    /// Forward pass dengan activation function
    ///
    /// # Arguments
    /// * `input` - Input matrix
    /// * `activation` - Activation function yang akan diterapkan
    ///
    /// # Returns
    /// Activated output matrix
    pub fn forward(&mut self, input: &Matrix, activation: &dyn ActivationFunction) -> Matrix {
        let raw_output = self.forward_raw(input);
        // Apply activation function ke setiap elemen
        raw_output.map(|x| activation.activate(x))
    }

    /// Backward pass - menghitung gradients untuk weights dan bias
    /// Ini adalah inti dari backpropagation algorithm
    ///
    /// # Arguments
    /// * `output_gradient` - Gradient dari layer selanjutnya [batch_size x output_size]
    /// * `activation` - Activation function yang digunakan di forward pass
    /// * `learning_rate` - Learning rate untuk update weights
    ///
    /// # Returns
    /// Input gradient untuk layer sebelumnya [batch_size x input_size]
    pub fn backward(
        &mut self,
        output_gradient: &Matrix,
        activation: &dyn ActivationFunction,
        learning_rate: f32,
    ) -> Matrix {
        let last_input = self
            .last_input
            .as_ref()
            .expect("Forward pass must be called before backward pass");
        let last_output = self
            .last_output
            .as_ref()
            .expect("Forward pass must be called before backward pass");

        // Step 1: Hitung activation gradient
        // activation_grad = output_gradient ⊙ activation'(last_output)
        let activation_grad =
            output_gradient.hadamard(&last_output.map(|x| activation.derivative(x)));

        // Step 2: Hitung weight gradients
        // weight_grad = input^T * activation_grad
        let weight_grad = last_input.transpose().multiply(&activation_grad);

        // Step 3: Hitung bias gradients
        // bias_grad = sum of activation_grad across batch dimension
        let mut bias_grad = Matrix::new(1, self.output_size);
        for j in 0..self.output_size {
            let mut sum = 0.0;
            for i in 0..activation_grad.rows {
                sum += activation_grad.get(i, j);
            }
            bias_grad.set(0, j, sum);
        }

        // Step 4: Update weights dan bias
        // weights = weights - learning_rate * weight_grad
        // bias = bias - learning_rate * bias_grad
        let weight_update = weight_grad.scale(learning_rate);
        let bias_update = bias_grad.scale(learning_rate);

        self.weights = self.weights.subtract(&weight_update);
        self.bias = self.bias.subtract(&bias_update);

        // Step 5: Hitung input gradient untuk layer sebelumnya
        // input_grad = activation_grad * weights^T
        activation_grad.multiply(&self.weights.transpose())
    }

    /// Get jumlah parameters (weights + bias) dalam layer ini
    pub fn parameter_count(&self) -> usize {
        self.weights.data.len() + self.bias.data.len()
    }
}
