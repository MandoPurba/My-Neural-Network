/// Matrix adalah structur data dasar untuk Neural Network (NN)
/// Setiap Layer dalam NN melakukan operasi matrix multiplication
#[derive(Debug, Clone)]
pub struct Matrix {
    pub data: Vec<f32>, // Data disimpan dalam 1D vector untuk efisiensi
    pub rows: usize,    // Jumlah baris
    pub cols: usize,    // Jumlah kolom
}

impl Matrix {
    /// Membuat matrix baru dengan ukuran tertentu, diisi dengan nol
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.; rows * cols],
            rows,
            cols,
        }
    }

    /// Membuat matrix dari data yang sudah ada
    pub fn from_data(data: Vec<f32>, rows: usize, cols: usize) -> Self {
        assert_eq!(
            data.len(),
            rows * cols,
            "Data length must be match matrix dimension"
        );
        Self { data, rows, cols }
    }

    /// Membuat metrix dengan nilai random (untuk inisialisi weights)
    /// Menunakan Xavier initialization: random values between -1/sqrt(n) and 1/sqrt(n)
    pub fn random(rows: usize, cols: usize) -> Self {
        let limit = 1. / (cols as f32).sqrt();
        let mut data = Vec::with_capacity(rows * cols);

        // Simple pseudo-random generator untuk demo
        let mut seed = 12345u32;
        for _ in 0..(rows * cols) {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let rand_val = (seed as f32 / u32::MAX as f32) * 2. - 1.;
            data.push(rand_val * limit);
        }
        Self { data, rows, cols }
    }

    /// Mengakses elemen matrix pada posisi  (row, col)
    pub fn get(&self, row: usize, col: usize) -> f32 {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col]
    }

    /// Mengubah nilai elemen matrix pada posisi (row, col)
    pub fn set(&mut self, row: usize, col: usize, value: f32) {
        assert!(row < self.rows && col < self.cols, "Index out of bounds");
        self.data[row * self.cols + col] = value;
    }

    /// Matrix multiplication - INTI dari Neural Network!
    /// Ini adalah operasi paling penting dalam NN
    /// Setiap layer melakukan: output = input * weights
    pub fn multiply(&self, other: &Matrix) -> Matrix {
        assert_eq!(
            self.cols, other.rows,
            "Matrix dimension don't match for multipication: {}x{} * {}x{}",
            self.rows, self.cols, other.rows, other.cols
        );

        let mut result = Matrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.;
                for k in 0..self.cols {
                    sum += self.get(i, k) * other.get(k, j);
                }
                result.set(i, j, sum);
            }
        }
        result
    }

    /// Element-wise addition (matrix + matrix)
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix rows must match");
        assert_eq!(self.cols, other.cols, "Matrix cols must match");

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] + other.data[i];
        }
        result
    }

    /// Element-wise subtraction (matrix - matrix)
    pub fn subtract(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix rows must match");
        assert_eq!(self.cols, other.cols, "Matrix cols must match");

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] - other.data[i];
        }
        result
    }

    /// Scalar multiplication (matrix * scalar)
    pub fn scale(&self, scalar: f32) -> Matrix {
        let mut result = self.clone();
        for val in &mut result.data {
            *val *= scalar;
        }
        result
    }

    /// Transpose matrix (menukar rows dan columns)
    /// Penting untuk backpropagation
    /// Contoh: jika matrix awal 2x3, setelah transpose menjadi 3x2
    /// Digunakan untuk menghitung gradient dengan benar
    /// Contoh: jika W adalah weights matrix 3x2, maka W^T adalah 2x3
    /// Digunakan dalam perhitungan gradient: dL/dX = dL/dY * W^T
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.set(j, i, self.get(i, j));
            }
        }
        result
    }

    /// Apply function ke setiap elemen (untuk actication functions)
    /// Misal ReLU, Sigmoid, dsb
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f32) -> f32,
    {
        let mut result = self.clone();
        for val in &mut result.data {
            *val = f(*val);
        }
        result
    }

    /// Hadamard product (element-wise multiplication)
    /// Berguna untuk backpropagation
    pub fn hadamard(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.rows, other.rows, "Matrix rows must match");
        assert_eq!(self.cols, other.cols, "Matrix cols must match");

        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.data.len() {
            result.data[i] = self.data[i] * other.data[i];
        }
        result
    }

    /// Print matrix untuk debugging
    pub fn print(&self) {
        println!("Matrix {}x{}:", self.rows, self.cols);
        for i in 0..self.rows {
            print!("[");
            for j in 0..self.cols {
                print!("{:8.4}", self.get(i, j));
                if j < self.cols - 1 {
                    print!(", ");
                }
            }
            println!("]");
        }
        println!();
    }
}
