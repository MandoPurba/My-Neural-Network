# Mini-Burn Deep Learning Framework

[![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Mini-Burn adalah implementasi minimal dari framework deep learning yang terinspirasi oleh [Burn](https://burn.dev), dibangun dengan Rust untuk tujuan pembelajaran. Framework ini mendemonstrasikan konsep-konsep kunci dalam membangun sistem tensor yang type-safe dan performant.

## 🚀 Fitur Utama

- **Type-Safe Tensors**: System tensor dengan type safety yang kuat menggunakan generics
- **Multi-Backend Support**: Abstraksi backend untuk berbagai hardware (saat ini mendukung CPU)
- **Multi-Dimensional Support**: Tensor dengan dimensi yang di-track pada compile time
- **Multiple Data Types**: Mendukung Float, Int, dan Bool tensor
- **Zero Dependencies**: Dibangun dari scratch tanpa dependencies eksternal
- **Comprehensive Operations**: Operasi aritmatika, matrix multiplication, fungsi matematika
- **Activation Functions**: ReLU, Sigmoid, Tanh, Softmax, GELU, Swish, ELU, Leaky ReLU
- **Computer Vision**: Neural network untuk image classification (MNIST-like examples)
- **Deep Learning**: Multi-layer networks dengan forward pass computation

## 📖 Arsitektur

```
Tensor<B, D, K>
├── B: Backend (CpuBackend, GpuBackend, etc.)
├── D: Dimensions (compile-time constant)
└── K: Data Type (Float, Int, Bool)
```

### Contoh Penggunaan Dasar

```rust
use mini_burn::{Tensor, CpuBackend, Shape, Float, Int, Bool};

// Tensor default (Float)
let tensor: Tensor<CpuBackend, 2> = 
    Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));

// Tensor dengan tipe eksplisit
let float_tensor: Tensor<CpuBackend, 2, Float> = 
    Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));

let int_tensor: Tensor<CpuBackend, 2, Int> = 
    Tensor::from_data(vec![1, 2, 3, 4], Shape::new([2, 2]));

let bool_tensor: Tensor<CpuBackend, 2, Bool> = 
    Tensor::from_data(vec![true, false, true, false], Shape::new([2, 2]));
```

## 🔥 Operasi Aktivasi

Framework ini mendukung berbagai fungsi aktivasi yang umum digunakan dalam deep learning:

```rust
let x: Tensor<CpuBackend, 1> = 
    Tensor::from_data(vec![-2.0, -1.0, 0.0, 1.0, 2.0], Shape::new([5]));

// Fungsi aktivasi dasar
let relu_output = x.relu();               // ReLU: max(0, x)
let sigmoid_output = x.sigmoid();         // Sigmoid: 1/(1 + exp(-x))
let tanh_output = x.tanh();              // Tanh: tanh(x)

// Fungsi aktivasi lanjutan
let leaky_relu_output = x.leaky_relu(0.1); // Leaky ReLU dengan alpha=0.1
let gelu_output = x.gelu();              // GELU (Gaussian Error Linear Unit)
let swish_output = x.swish();            // Swish/SiLU: x * sigmoid(x)
let elu_output = x.elu(1.0);             // ELU dengan alpha=1.0

// Softmax untuk probabilitas
let logits: Tensor<CpuBackend, 1> = 
    Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([4]));
let probabilities = logits.softmax();    // Output: distribusi probabilitas

// Softmax 2D (batch processing)
let batch_logits: Tensor<CpuBackend, 2> = 
    Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3]));
let batch_probs = batch_logits.softmax(); // Softmax per baris
```

### Karakteristik Fungsi Aktivasi

| Fungsi | Range Output | Kegunaan Utama |
|--------|-------------|----------------|
| ReLU | [0, ∞) | Hidden layers, cepat dan sederhana |
| Leaky ReLU | (-∞, ∞) | Mengatasi dying ReLU problem |
| Sigmoid | (0, 1) | Binary classification, output layer |
| Tanh | (-1, 1) | Hidden layers, zero-centered |
| Softmax | (0, 1), sum=1 | Multi-class classification |
| GELU | (-∞, ∞) | Modern transformer models |
| Swish/SiLU | (-∞, ∞) | Smooth, self-gated activation |
| ELU | (-α, ∞) | Smooth negative part |

## 🧠 Neural Network Example

```rust
use mini_burn::{Tensor, CpuBackend, Shape};

// Simulasi forward pass pada dense layer
let input: Tensor<CpuBackend, 2> = 
    Tensor::from_data(vec![0.5, -0.2, 1.0, -0.1, 0.8, -0.5], Shape::new([2, 3]));

let weights: Tensor<CpuBackend, 2> = 
    Tensor::from_data(vec![0.2, -0.1, 0.3, 0.4, 0.1, 0.5, -0.2, 0.0, -0.3, 0.2, 0.1, 0.6], 
                     Shape::new([3, 4]));

// Forward pass: output = input @ weights
let linear_output = input.matmul(&weights);

// Aplikasi aktivasi yang berbeda
let relu_output = linear_output.relu();
let softmax_output = linear_output.softmax(); // Probabilitas per sample
```

## 🛠️ Instalasi

Tambahkan ke `Cargo.toml`:

```toml
[dependencies]
mini_burn = "0.1.0"
```

Atau clone repository ini:

```bash
git clone https://github.com/yourusername/mini_burn.git
cd mini_burn
cargo build --release
```

## 🚀 Quick Start

```rust
use mini_burn::{Tensor, CpuBackend, Shape};

fn main() {
    // Buat tensor 2D
    let a: Tensor<CpuBackend, 2> = 
        Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
    
    let b: Tensor<CpuBackend, 2> = 
        Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2]));
    
    // Operasi dasar
    let sum = a.add(&b);
    let product = a.mul(&b);
    let matrix_mult = a.matmul(&b);
    
    // Fungsi matematika
    let exp_a = a.exp();
    let sqrt_a = a.sqrt();
    
    // Fungsi aktivasi
    let relu_a = a.relu();
    let sigmoid_a = a.sigmoid();
    let softmax_a = a.softmax();
    
    println!("Sum: {:?}", sum.to_data());
    println!("ReLU: {:?}", relu_a.to_data());
    println!("Softmax: {:?}", softmax_a.to_data());
}
```

## 📊 Examples

Jalankan contoh-contoh yang tersedia:

```bash
# Penggunaan dasar tensor dan operasi
cargo run --example basic_usage

# Operasi matematika lanjutan
cargo run --example advanced_operations

# Demonstrasi fungsi aktivasi
cargo run --example activation_functions

# Simulasi neural network sederhana
cargo run --example simple_neural_network

# Computer vision - MNIST classification demo
cargo run --example mnist_demo

# Simple MNIST - pattern recognition
cargo run --example simple_mnist

# Test performa
cargo run --example performance_test --release
```

## 🧪 Testing

```bash
# Jalankan semua test
cargo test

# Test spesifik aktivasi function
cargo test activation

# Test dengan output verbose
cargo test -- --nocapture
```

## 📚 Struktur Proyek

```
mini_burn/
├── src/
│   ├── lib.rs              # Entry point dan re-exports
│   ├── backend/            # Backend abstraction
│   │   └── mod.rs         # CPU backend implementation
│   ├── data/              # Data type system
│   │   └── mod.rs         # Float, Int, Bool types
│   ├── shape/             # Shape management
│   │   └── mod.rs         # Multi-dimensional shapes
│   ├── tensor/            # Core tensor implementation
│   │   └── mod.rs         # Tensor struct dan factory methods
│   └── ops/               # Tensor operations
│       └── mod.rs         # Arithmetic, math, activations
├── examples/              # Comprehensive examples
│   ├── basic_usage.rs
│   ├── activation_functions.rs
│   ├── simple_neural_network.rs
│   ├── mnist_demo.rs      # Computer vision demo
│   ├── simple_mnist.rs    # Pattern recognition
│   ├── advanced_operations.rs
│   └── performance_test.rs
├── tests/
│   └── integration_tests.rs
└── docs/
    ├── ARCHITECTURE.md
    └── CHANGELOG.md
```

## 🎯 Roadmap

### Versi Selanjutnya
- [ ] **Automatic Differentiation**: Backward pass untuk training
- [ ] **Broadcasting**: Operasi tensor dengan shape yang berbeda
- [ ] **GPU Backend**: Support CUDA/OpenCL
- [ ] **SIMD Optimizations**: Vectorized operations
- [ ] **More Data Types**: f64, i64, bfloat16
- [ ] **Memory Pool**: Efficient memory management
- [ ] **Model APIs**: Layers, optimizers, loss functions

### Optimizations
- [ ] **Blocked Matrix Multiplication**: Efficient GEMM
- [ ] **Multi-threading**: Parallel operations dengan Rayon
- [ ] **In-place Operations**: Reduce memory allocations
- [ ] **JIT Compilation**: Runtime optimization

## 🔬 Architecture Deep Dive

### Type System

Mini-Burn menggunakan sistem type yang kuat untuk memastikan safety:

```rust
// B: Backend type (determines where computation happens)
// D: Dimensions (compile-time constant)
// K: Data type (Float, Int, Bool with default Float)
pub struct Tensor<B: Backend, const D: usize, K: DataType = Float>

// Contoh spesifikasi eksplisit
let tensor: Tensor<CpuBackend, 2, Float> = // 2D float tensor di CPU
let int_tensor: Tensor<CpuBackend, 3, Int> = // 3D int tensor di CPU
```

### Backend Abstraction

```rust
pub trait Backend {
    type Device: Device;
    type Storage<T: DataType>: Storage<T>;
}

// Implementasi CPU
pub struct CpuBackend;
pub struct CpuDevice;
pub struct CpuStorage<T: DataType> {
    data: Vec<T::Primitive>,
}
```

### Shape System

```rust
pub struct Shape<const D: usize> {
    dims: [usize; D],
}

// Compile-time shape verification
let shape_2d: Shape<2> = Shape::new([3, 4]); // ✅ OK
let shape_3d: Shape<3> = Shape::new([2, 3, 4]); // ✅ OK
// let invalid: Shape<2> = Shape::new([2, 3, 4]); // ❌ Compile error
```

## 🤝 Contributing

Kontribusi sangat diterima! Beberapa area yang bisa dikontribusikan:

1. **Performance**: Optimasi operasi matematik
2. **Features**: Implementasi operasi baru
3. **Documentation**: Perbaikan dokumentasi dan contoh
4. **Testing**: Menambah coverage test
5. **Examples**: Contoh penggunaan yang lebih kompleks

### Development Setup

```bash
git clone https://github.com/yourusername/mini_burn.git
cd mini_burn
cargo build
cargo test
cargo run --example basic_usage
```

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🎯 MNIST Computer Vision Examples

Framework ini dilengkapi dengan dua contoh computer vision:

### 1. Full MNIST Demo (`mnist_demo.rs`)
```bash
cargo run --example mnist_demo
```
- Simulasi MNIST 28x28 images dengan 10 classes (digits 0-9)
- Multi-layer neural network (784 → 128 → 64 → 32 → 10)
- Batch processing dan evaluation metrics
- ASCII visualization dari synthetic images
- Demonstrasi comprehensive untuk computer vision

### 2. Simple MNIST (`simple_mnist.rs`)
```bash
cargo run --example simple_mnist
```
- Pattern recognition dengan 8x8 images
- 3 classes sederhana: Circle, Cross, Square
- Neural network yang lebih kecil (64 → 32 → 16 → 3)
- Mudah dipahami untuk pembelajaran
- Menunjukkan konsep dasar classification

## 🧠 Computer Vision Capabilities

```rust
// Contoh neural network untuk image classification
let input: Tensor<CpuBackend, 2> = // Batch of images [batch_size, pixels]
    Tensor::from_data(image_data, Shape::new([8, 784]));

// Forward pass
let hidden1 = input.matmul(&weights1).relu();
let hidden2 = hidden1.matmul(&weights2).relu();
let output = hidden2.matmul(&weights3).softmax(); // Probabilities per class

// Classification
let predicted_class = find_max_index(&output.to_data());
```

### Features Demonstrated:
- ✅ Image preprocessing (flattening 2D images to 1D vectors)
- ✅ Multi-layer dense networks
- ✅ ReLU activations untuk hidden layers
- ✅ Softmax untuk multi-class classification
- ✅ Batch processing untuk efficiency
- ✅ Synthetic data generation
- ✅ Model evaluation dan accuracy metrics
- ✅ ASCII art visualization

## 🙏 Acknowledgments

- Terinspirasi oleh [Burn](https://burn.dev) - Modern deep learning framework
- Rust community untuk ecosystem yang luar biasa
- PyTorch dan TensorFlow untuk referensi API design
- MNIST dataset concept untuk computer vision examples

---

**Note**: Ini adalah framework untuk pembelajaran dan eksperimen. Untuk production, gunakan framework yang sudah mature seperti Burn, Candle, atau tch.