# 🧠 Mini Neural Network Framework

Sebuah implementasi Neural Network dari scratch menggunakan Rust untuk tujuan pembelajaran. Framework ini dibuat untuk memahami bagaimana Neural Network bekerja secara fundamental tanpa menggunakan library deep learning yang kompleks.

## 🎯 Tujuan Proyek

- **Pembelajaran**: Memahami cara kerja Neural Network dari dasar
- **Implementasi Manual**: Semua operasi matrix dan algoritma dibuat dari scratch
- **Sederhana tapi Lengkap**: Code yang mudah dipahami namun tetap fungsional
- **Bahasa Indonesia**: Komentar dan dokumentasi dalam bahasa Indonesia

## 🚀 Fitur

### ✅ Operasi Matrix
- Matrix multiplication, addition, subtraction
- Transpose, Hadamard product (element-wise multiplication)
- Scalar operations dan mapping functions
- Xavier weight initialization

### ✅ Activation Functions
- **ReLU**: `f(x) = max(0, x)` - Paling populer untuk hidden layers
- **Sigmoid**: `f(x) = 1/(1 + e^(-x))` - Bagus untuk binary classification
- **Tanh**: `f(x) = tanh(x)` - Centered around zero
- **Linear**: `f(x) = x` - Untuk regression output
- **Softmax**: Untuk multi-class classification

### ✅ Loss Functions
- **Mean Squared Error (MSE)**: Untuk regression problems
- **Binary Cross Entropy**: Untuk binary classification

### ✅ Neural Network
- Dense/Fully Connected Layers
- Forward propagation
- Backpropagation algorithm
- Gradient descent optimization
- Batch training support
- Training statistics tracking

## 📊 Struktur Proyek

```
My-Neural-Network/
├── src/
│   ├── lib.rs          # Module exports
│   ├── matrix.rs       # Matrix operations
│   ├── activation.rs   # Activation functions
│   ├── loss.rs         # Loss functions
│   ├── layer.rs        # Dense layer implementation
│   ├── network.rs      # Neural network implementation
│   └── main.rs         # Demo dan examples
├── Cargo.toml
└── README.md
```

## 🔧 Instalasi dan Menjalankan

```bash
# Clone repository
git clone <repository-url>
cd My-Neural-Network

# Compile dan run demo
cargo run --bin demo

# Atau build saja
cargo build --release
```

## 📖 Cara Penggunaan

### 1. Basic Matrix Operations

```rust
use mini_burn::*;

// Buat matrix
let a = Matrix::from_data(vec![1.0, 2.0, 3.0, 4.0], 2, 2);
let b = Matrix::from_data(vec![5.0, 6.0, 7.0, 8.0], 2, 2);

// Operasi matrix
let c = a.multiply(&b);     // Matrix multiplication
let d = a.add(&b);          // Element-wise addition
let e = a.transpose();      // Transpose
```

### 2. Membuat Neural Network

```rust
// Cara 1: Manual layer by layer
let mut network = NeuralNetwork::new();
network.add_dense_layer(784, 128, "relu");    // Input layer
network.add_dense_layer(128, 64, "relu");     // Hidden layer
network.add_dense_layer(64, 10, "sigmoid");   // Output layer

// Cara 2: Architecture builder (lebih mudah)
let mut network = NeuralNetwork::from_architecture(
    &[784, 128, 64, 10],  // Layer sizes
    "relu",               // Hidden activation
    "sigmoid"             // Output activation
);
```

### 3. Training Network

```rust
// Siapkan data
let train_inputs = Matrix::from_data(/* your input data */, rows, cols);
let train_targets = Matrix::from_data(/* your target data */, rows, cols);

// Pilih loss function
let loss_fn = BinaryCrossEntropy;  // atau MeanSquaredError

// Training
network.train(
    &train_inputs,    // Training data
    &train_targets,   // Target labels
    &loss_fn,        // Loss function
    0.01,            // Learning rate
    1000,            // Epochs
    true             // Verbose output
);
```

### 4. Prediksi

```rust
// Prediksi pada data baru
let test_input = Matrix::from_data(vec![0.5, -0.3, 1.2], 1, 3);
let prediction = network.predict(&test_input);

// Evaluasi accuracy (untuk classification)
let accuracy = network.calculate_accuracy(&test_inputs, &test_targets);
println!("Accuracy: {:.2}%", accuracy * 100.0);
```

## 🎲 Contoh: XOR Problem

XOR adalah problem klasik yang tidak bisa diselesaikan dengan linear classifier, sehingga butuh Neural Network:

```rust
fn demo_xor() {
    // Data XOR: [x1, x2] -> [x1 XOR x2]
    let inputs = Matrix::from_data(vec![
        0.0, 0.0,  // 0 XOR 0 = 0
        0.0, 1.0,  // 0 XOR 1 = 1
        1.0, 0.0,  // 1 XOR 0 = 1
        1.0, 1.0,  // 1 XOR 1 = 0
    ], 4, 2);
    
    let targets = Matrix::from_data(vec![0.0, 1.0, 1.0, 0.0], 4, 1);
    
    // Network: 2 input -> 4 hidden -> 1 output
    let mut network = NeuralNetwork::from_architecture(
        &[2, 4, 1], "relu", "sigmoid"
    );
    
    // Training
    let loss_fn = BinaryCrossEntropy;
    network.train(&inputs, &targets, &loss_fn, 0.1, 1000, true);
    
    // Test
    let predictions = network.predict(&inputs);
    // Network seharusnya bisa memprediksi XOR dengan benar!
}
```

## 📚 Konsep Neural Network yang Diimplementasi

### 1. **Forward Propagation**
```
Input -> Layer1 -> Activation -> Layer2 -> Activation -> ... -> Output
```

Setiap layer melakukan operasi: `output = activation(input × weights + bias)`

### 2. **Backpropagation**
```
Loss -> ∂Loss/∂Output -> ∂Loss/∂Weights -> Update Weights
```

Algorithm untuk menghitung gradients dan update weights:
- Hitung error di output layer
- Propagate error mundur ke layer sebelumnya  
- Update weights menggunakan gradient descent

### 3. **Gradient Descent**
```
weights_new = weights_old - learning_rate × gradient
```

### 4. **Matrix Operations yang Penting**
- **Forward**: `Y = X × W + B`
- **Weight Gradient**: `∂W = X^T × ∂Y`
- **Input Gradient**: `∂X = ∂Y × W^T`

## 🎯 Problem yang Bisa Diselesaikan

### Classification Problems
- Binary classification (sigmoid output + binary cross entropy)
- Multi-class classification (softmax output + categorical cross entropy)

### Regression Problems  
- Function approximation (linear output + MSE loss)
- Curve fitting

### Demo yang Tersedia
- **XOR Problem**: Classic non-linear classification
- **Simple Regression**: y = 2x + 1 function approximation

## 🔍 Detail Implementasi

### Matrix Operations (`matrix.rs`)
- Storage: 1D vector untuk efisiensi memory
- Row-major order untuk akses yang cepat
- Bounds checking untuk safety

### Dense Layer (`layer.rs`)
- Weights: Xavier initialization untuk konvergensi yang baik
- Bias: Initialized to zero
- Forward: Linear transformation + activation
- Backward: Gradient computation + weight updates

### Activation Functions (`activation.rs`)
- Trait-based design untuk extensibility
- Forward pass: `activate(x)`
- Backward pass: `derivative(x)`

### Loss Functions (`loss.rs`)
- Trait-based design
- Forward: `calculate_loss(predictions, targets)`
- Backward: `calculate_gradient(predictions, targets)`

## 🚧 Limitasi Saat Ini

- Hanya Dense/Fully Connected layers (belum ada CNN, RNN)
- Optimizer hanya basic gradient descent (belum ada Adam, RMSprop)
- Belum ada regularization (dropout, L1/L2)
- Belum ada batch normalization
- Single-threaded (belum ada parallelization)

## 🎓 Tujuan Pembelajaran

Framework ini dibuat untuk memahami:

1. **Bagaimana matrix operations bekerja** dalam deep learning
2. **Algoritma backpropagation** step by step
3. **Gradient descent optimization** secara manual
4. **Architecture design** untuk neural networks
5. **Forward dan backward pass** dalam detail

## 🤝 Kontribusi

Silakan berkontribusi untuk:
- Menambah activation functions baru
- Implementasi optimizer yang lebih advanced
- Menambah layer types (Convolutional, LSTM, dll)
- Optimisasi performance
- Menambah examples dan tutorials

## 📜 Lisensi

MIT License - Bebas digunakan untuk pembelajaran dan development.

## 🙏 Acknowledgments

Inspired by:
- Neural Networks and Deep Learning by Michael Nielsen
- Deep Learning by Ian Goodfellow
- Rust programming language community

---

**Catatan**: Ini adalah implementasi untuk pembelajaran. Untuk production, gunakan framework seperti Candle, tch, atau PyTorch.