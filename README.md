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

## 📖 Arsitektur

```
Tensor<B, D, K>
├── B: Backend (CpuBackend, GpuBackend, etc.)
├── D: Dimensions (compile-time constant)
└── K: Data Type (Float, Int, Bool)
```

### Contoh Penggunaan

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

## 📚 Dokumentasi

### Factory Methods

```rust
// Tensor nol
let zeros = Tensor::<CpuBackend, 2>::zeros(Shape::new([3, 3]));

// Tensor satu
let ones = Tensor::<CpuBackend, 2>::ones(Shape::new([3, 3]));

// Tensor dengan value tertentu
let filled = Tensor::<CpuBackend, 2>::full(Shape::new([2, 2]), 42.0);

// Range tensor
let range = Tensor::<CpuBackend, 1>::range(0.0, 10.0, 2.0); // [0, 2, 4, 6, 8]
```

### Operasi Tensor

```rust
let a = Tensor::from_data(vec![1.0, 2.0, 3.0, 4.0], Shape::new([2, 2]));
let b = Tensor::from_data(vec![5.0, 6.0, 7.0, 8.0], Shape::new([2, 2]));

// Element-wise operations
let sum = a.add(&b);        // [6, 8, 10, 12]
let diff = a.sub(&b);       // [-4, -4, -4, -4]
let prod = a.mul(&b);       // [5, 12, 21, 32]
let div = a.div(&b);        // [0.2, 0.33, 0.43, 0.5]

// Scalar operations
let scalar_add = a.add_scalar(10.0); // [11, 12, 13, 14]

// Matrix multiplication
let matmul = a.matmul(&b);  // Matrix multiplication untuk 2D tensors

// Mathematical functions
let sin_result = a.sin();
let cos_result = a.cos();
let exp_result = a.exp();
let sqrt_result = a.sqrt();

// Reduction operations
let sum_all = a.sum();      // Sum semua elemen
let mean_val = a.mean();    // Rata-rata
let max_val = a.max();      // Nilai maksimum
let min_val = a.min();      // Nilai minimum
```

### Multi-Dimensional Tensors

```rust
// 1D vector
let vector: Tensor<CpuBackend, 1> = 
    Tensor::from_data(vec![1.0, 2.0, 3.0], Shape::new([3]));

// 2D matrix
let matrix: Tensor<CpuBackend, 2> = 
    Tensor::from_data(vec![1.0; 6], Shape::new([2, 3]));

// 3D tensor
let tensor_3d: Tensor<CpuBackend, 3> = 
    Tensor::from_data(vec![1.0; 24], Shape::new([2, 3, 4]));

// 4D tensor (common in deep learning)
let tensor_4d: Tensor<CpuBackend, 4> = 
    Tensor::from_data(vec![1.0; 120], Shape::new([2, 3, 4, 5]));
```

## 🧪 Menjalankan Examples

Framework dilengkapi dengan beberapa contoh penggunaan:

```bash
# Penggunaan dasar
cargo run --example basic_usage

# Operasi advanced
cargo run --example advanced_operations

# Neural network sederhana
cargo run --example simple_neural_network

# Performance testing
cargo run --example performance_test
```

## 🧪 Testing

Jalankan semua tests:

```bash
cargo test
```

Jalankan tests dengan output detail:

```bash
cargo test -- --nocapture
```

## 📊 Struktur Proyek

```
mini_burn/
├── src/
│   ├── lib.rs              # Entry point dan re-exports
│   ├── backend/mod.rs      # Backend abstraction (CPU)
│   ├── tensor/mod.rs       # Core tensor implementation
│   ├── data/mod.rs         # Data types (Float, Int, Bool)
│   ├── shape/mod.rs        # Shape dan dimension tracking
│   └── ops/mod.rs          # Tensor operations
├── examples/
│   ├── basic_usage.rs      # Contoh penggunaan dasar
│   ├── advanced_operations.rs
│   ├── simple_neural_network.rs
│   └── performance_test.rs
├── tests/
│   └── integration_tests.rs
└── README.md
```

## 🎯 Contoh Neural Network

```rust
use mini_burn::{Tensor, CpuBackend, Shape};

// Implementasi layer linear sederhana
struct LinearLayer {
    weights: Tensor<CpuBackend, 2>,
    bias: Tensor<CpuBackend, 1>,
}

impl LinearLayer {
    fn new(input_size: usize, output_size: usize) -> Self {
        let weights = Tensor::zeros(Shape::new([input_size, output_size]));
        let bias = Tensor::zeros(Shape::new([output_size]));
        Self { weights, bias }
    }
    
    fn forward(&self, input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
        // Simplified forward pass
        // input @ weights + bias
        // ... implementasi detail
    }
}

// Fungsi aktivasi
fn relu(input: &Tensor<CpuBackend, 1>) -> Tensor<CpuBackend, 1> {
    let data: Vec<f32> = input.to_data()
        .iter()
        .map(|&x| if x > 0.0 { x } else { 0.0 })
        .collect();
    
    Tensor::from_data(data, input.shape().clone())
}
```

## 🔬 Pembelajaran

Framework ini dibuat untuk tujuan pembelajaran deep learning dan Rust. Beberapa konsep yang dipelajari:

1. **Type System**: Penggunaan generics dan trait bounds untuk type safety
2. **Memory Management**: Rust ownership model untuk tensor data
3. **Abstraction**: Backend abstraction untuk portability
4. **Performance**: Zero-cost abstractions
5. **Architecture Design**: Sistem modular dan extensible

## 🚧 Roadmap

### Phase 1: Core Improvements
- [ ] Broadcasting support
- [ ] More tensor operations (sin, cos, exp, log, etc.) ✅
- [ ] Better error handling dengan Result types
- [ ] SIMD optimizations

### Phase 2: Deep Learning Features
- [ ] Automatic differentiation
- [ ] Neural network layers
- [ ] Optimizers (SGD, Adam)
- [ ] Loss functions

### Phase 3: Advanced Features
- [ ] GPU backend
- [ ] Model serialization
- [ ] Graph optimization
- [ ] JIT compilation

### Phase 4: Production Features
- [ ] Distributed training
- [ ] Mixed precision
- [ ] Dynamic shapes
- [ ] Custom kernels

## 📈 Performance

Current implementation:
- **Element-wise operations**: O(n) linear dengan jumlah elemen
- **Matrix multiplication**: O(n³) algoritma naive
- **Memory allocation**: Tensor baru dibuat untuk setiap operasi
- **SIMD**: Belum diimplementasikan

Optimization opportunities:
- SIMD instructions untuk operasi vectorized
- BLAS integration untuk linear algebra yang optimal
- In-place operations untuk mengurangi alokasi memori
- Parallel processing untuk tensor besar
- GPU acceleration

## 🤝 Kontribusi

Ini adalah proyek pembelajaran, silakan fork dan eksperimen sesuai kebutuhan Anda!

### Guidelines
1. Fork repository
2. Buat feature branch
3. Commit changes dengan descriptive messages
4. Tambahkan tests untuk fitur baru
5. Buat pull request

## 📝 Lisensi

Proyek ini dilisensikan under MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## 🙏 Acknowledgments

- Inspired by [Burn Framework](https://burn.dev)
- Rust community untuk dokumentasi dan resources yang excellent
- Deep learning community untuk knowledge sharing

## 📧 Kontak

Jika ada pertanyaan atau saran, silakan buat issue di repository ini.

---

**Note**: Ini adalah implementasi minimal untuk tujuan pembelajaran. Untuk production workloads, gunakan framework yang sudah matang seperti Burn, Candle, atau tch.