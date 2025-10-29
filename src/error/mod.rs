//! Error Handling Module
//!
//! This module provides comprehensive error handling for the Mini-Burn framework,
//! ensuring production reliability with proper error types, error chains, and
//! recovery mechanisms.

use std::fmt;

/// The main error type for the Mini-Burn framework
#[derive(Debug)]
pub enum MiniBurnError {
    /// Tensor operation errors
    Tensor(TensorError),
    /// Shape-related errors
    Shape(ShapeError),
    /// Backend-specific errors
    Backend(BackendError),
    /// Automatic differentiation errors
    Autodiff(AutodiffError),
    /// Optimization errors
    Optimizer(OptimizerError),
    /// Training errors
    Training(TrainingError),
    /// Broadcasting errors
    Broadcasting(crate::broadcast::BroadcastError),
    /// Neural network errors
    Network(NetworkError),
    /// Data loading/processing errors
    Data(DataError),
    /// Memory allocation errors
    Memory(MemoryError),
    /// Configuration errors
    Config(ConfigError),
    /// I/O errors
    Io(String),
    /// General framework error
    General(String),
}

impl fmt::Display for MiniBurnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MiniBurnError::Tensor(e) => write!(f, "Tensor error: {}", e),
            MiniBurnError::Shape(e) => write!(f, "Shape error: {}", e),
            MiniBurnError::Backend(e) => write!(f, "Backend error: {}", e),
            MiniBurnError::Autodiff(e) => write!(f, "Autodiff error: {}", e),
            MiniBurnError::Optimizer(e) => write!(f, "Optimizer error: {}", e),
            MiniBurnError::Training(e) => write!(f, "Training error: {}", e),
            MiniBurnError::Broadcasting(e) => write!(f, "Broadcasting error: {}", e),
            MiniBurnError::Network(e) => write!(f, "Network error: {}", e),
            MiniBurnError::Data(e) => write!(f, "Data error: {}", e),
            MiniBurnError::Memory(e) => write!(f, "Memory error: {}", e),
            MiniBurnError::Config(e) => write!(f, "Config error: {}", e),
            MiniBurnError::Io(e) => write!(f, "I/O error: {}", e),
            MiniBurnError::General(msg) => write!(f, "Mini-Burn error: {}", msg),
        }
    }
}

impl std::error::Error for MiniBurnError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MiniBurnError::Tensor(e) => Some(e),
            MiniBurnError::Shape(e) => Some(e),
            MiniBurnError::Backend(e) => Some(e),
            MiniBurnError::Autodiff(e) => Some(e),
            MiniBurnError::Optimizer(e) => Some(e),
            MiniBurnError::Training(e) => Some(e),
            MiniBurnError::Broadcasting(e) => Some(e),
            MiniBurnError::Network(e) => Some(e),
            MiniBurnError::Data(e) => Some(e),
            MiniBurnError::Memory(e) => Some(e),
            MiniBurnError::Config(e) => Some(e),
            MiniBurnError::Io(_) => None,
            MiniBurnError::General(_) => None,
        }
    }
}

/// Tensor operation errors
#[derive(Debug, Clone)]
pub enum TensorError {
    /// Invalid tensor dimensions
    InvalidDimensions {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Shape mismatch in operations
    ShapeMismatch {
        lhs: Vec<usize>,
        rhs: Vec<usize>,
        operation: String,
    },
    /// Index out of bounds
    IndexOutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    },
    /// Invalid data type for operation
    InvalidDataType { expected: String, actual: String },
    /// Data length mismatch
    DataLengthMismatch { expected: usize, actual: usize },
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Division by zero
    DivisionByZero,
    /// Invalid value (NaN, infinity, etc.)
    InvalidValue(String),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::InvalidDimensions { expected, actual } => {
                write!(
                    f,
                    "Invalid dimensions: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            TensorError::ShapeMismatch {
                lhs,
                rhs,
                operation,
            } => {
                write!(f, "Shape mismatch in {}: {:?} vs {:?}", operation, lhs, rhs)
            }
            TensorError::IndexOutOfBounds { index, shape } => {
                write!(f, "Index {:?} out of bounds for shape {:?}", index, shape)
            }
            TensorError::InvalidDataType { expected, actual } => {
                write!(
                    f,
                    "Invalid data type: expected {}, got {}",
                    expected, actual
                )
            }
            TensorError::DataLengthMismatch { expected, actual } => {
                write!(
                    f,
                    "Data length mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            TensorError::UnsupportedOperation(op) => {
                write!(f, "Unsupported operation: {}", op)
            }
            TensorError::DivisionByZero => write!(f, "Division by zero"),
            TensorError::InvalidValue(msg) => write!(f, "Invalid value: {}", msg),
        }
    }
}

impl std::error::Error for TensorError {}

/// Shape-related errors
#[derive(Debug, Clone)]
pub enum ShapeError {
    /// Invalid shape dimensions
    InvalidShape(Vec<usize>),
    /// Cannot reshape to target shape
    CannotReshape { from: Vec<usize>, to: Vec<usize> },
    /// Dimension out of bounds
    DimensionOutOfBounds { dim: usize, max: usize },
    /// Empty shape not allowed
    EmptyShape,
    /// Incompatible shapes for operation
    IncompatibleShapes {
        shapes: Vec<Vec<usize>>,
        operation: String,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::InvalidShape(shape) => write!(f, "Invalid shape: {:?}", shape),
            ShapeError::CannotReshape { from, to } => {
                write!(f, "Cannot reshape from {:?} to {:?}", from, to)
            }
            ShapeError::DimensionOutOfBounds { dim, max } => {
                write!(f, "Dimension {} out of bounds (max: {})", dim, max)
            }
            ShapeError::EmptyShape => write!(f, "Empty shape not allowed"),
            ShapeError::IncompatibleShapes { shapes, operation } => {
                write!(f, "Incompatible shapes for {}: {:?}", operation, shapes)
            }
        }
    }
}

impl std::error::Error for ShapeError {}

/// Backend-specific errors
#[derive(Debug, Clone)]
pub enum BackendError {
    /// Backend not available
    NotAvailable(String),
    /// Backend initialization failed
    InitializationFailed(String),
    /// Device not found
    DeviceNotFound(String),
    /// Operation not supported by backend
    UnsupportedOperation { backend: String, operation: String },
    /// Memory allocation failed
    AllocationFailed { size: usize, backend: String },
    /// Computation failed
    ComputationFailed(String),
    /// Backend-specific error
    BackendSpecific(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::NotAvailable(backend) => write!(f, "Backend not available: {}", backend),
            BackendError::InitializationFailed(msg) => {
                write!(f, "Backend initialization failed: {}", msg)
            }
            BackendError::DeviceNotFound(device) => write!(f, "Device not found: {}", device),
            BackendError::UnsupportedOperation { backend, operation } => {
                write!(
                    f,
                    "Operation {} not supported by backend {}",
                    operation, backend
                )
            }
            BackendError::AllocationFailed { size, backend } => {
                write!(
                    f,
                    "Memory allocation of {} bytes failed on backend {}",
                    size, backend
                )
            }
            BackendError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
            BackendError::BackendSpecific(msg) => write!(f, "Backend error: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Automatic differentiation errors
#[derive(Debug, Clone)]
pub enum AutodiffError {
    /// Gradient computation failed
    GradientComputationFailed(String),
    /// No gradient available
    NoGradientAvailable { node_id: String },
    /// Gradient shape mismatch
    GradientShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Computation graph error
    GraphError(String),
    /// Tape operation failed
    TapeOperationFailed(String),
    /// Backward pass failed
    BackwardPassFailed(String),
    /// Invalid gradient
    InvalidGradient(String),
}

impl fmt::Display for AutodiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AutodiffError::GradientComputationFailed(msg) => {
                write!(f, "Gradient computation failed: {}", msg)
            }
            AutodiffError::NoGradientAvailable { node_id } => {
                write!(f, "No gradient available for node: {}", node_id)
            }
            AutodiffError::GradientShapeMismatch { expected, actual } => {
                write!(
                    f,
                    "Gradient shape mismatch: expected {:?}, got {:?}",
                    expected, actual
                )
            }
            AutodiffError::GraphError(msg) => write!(f, "Computation graph error: {}", msg),
            AutodiffError::TapeOperationFailed(msg) => write!(f, "Tape operation failed: {}", msg),
            AutodiffError::BackwardPassFailed(msg) => write!(f, "Backward pass failed: {}", msg),
            AutodiffError::InvalidGradient(msg) => write!(f, "Invalid gradient: {}", msg),
        }
    }
}

impl std::error::Error for AutodiffError {}

/// Optimization errors
#[derive(Debug, Clone)]
pub enum OptimizerError {
    /// Parameter update failed
    ParameterUpdateFailed(String),
    /// Invalid learning rate
    InvalidLearningRate(f32),
    /// No parameters to optimize
    NoParameters,
    /// Convergence failed
    ConvergenceFailed { iterations: usize, tolerance: f32 },
    /// Invalid optimizer configuration
    InvalidConfiguration(String),
    /// Numerical instability
    NumericalInstability(String),
}

impl fmt::Display for OptimizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerError::ParameterUpdateFailed(msg) => {
                write!(f, "Parameter update failed: {}", msg)
            }
            OptimizerError::InvalidLearningRate(lr) => {
                write!(f, "Invalid learning rate: {}", lr)
            }
            OptimizerError::NoParameters => write!(f, "No parameters to optimize"),
            OptimizerError::ConvergenceFailed {
                iterations,
                tolerance,
            } => {
                write!(
                    f,
                    "Convergence failed after {} iterations (tolerance: {})",
                    iterations, tolerance
                )
            }
            OptimizerError::InvalidConfiguration(msg) => {
                write!(f, "Invalid optimizer configuration: {}", msg)
            }
            OptimizerError::NumericalInstability(msg) => {
                write!(f, "Numerical instability: {}", msg)
            }
        }
    }
}

impl std::error::Error for OptimizerError {}

/// Training errors
#[derive(Debug, Clone)]
pub enum TrainingError {
    /// Model forward pass failed
    ForwardPassFailed(String),
    /// Loss computation failed
    LossComputationFailed(String),
    /// Backward pass failed
    BackwardPassFailed(String),
    /// Validation failed
    ValidationFailed(String),
    /// Data loading failed
    DataLoadingFailed(String),
    /// Training interrupted
    TrainingInterrupted(String),
    /// Invalid training configuration
    InvalidConfiguration(String),
    /// Checkpoint loading/saving failed
    CheckpointError(String),
}

impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrainingError::ForwardPassFailed(msg) => write!(f, "Forward pass failed: {}", msg),
            TrainingError::LossComputationFailed(msg) => {
                write!(f, "Loss computation failed: {}", msg)
            }
            TrainingError::BackwardPassFailed(msg) => write!(f, "Backward pass failed: {}", msg),
            TrainingError::ValidationFailed(msg) => write!(f, "Validation failed: {}", msg),
            TrainingError::DataLoadingFailed(msg) => write!(f, "Data loading failed: {}", msg),
            TrainingError::TrainingInterrupted(msg) => write!(f, "Training interrupted: {}", msg),
            TrainingError::InvalidConfiguration(msg) => {
                write!(f, "Invalid training configuration: {}", msg)
            }
            TrainingError::CheckpointError(msg) => write!(f, "Checkpoint error: {}", msg),
        }
    }
}

impl std::error::Error for TrainingError {}

/// Neural network errors
#[derive(Debug, Clone)]
pub enum NetworkError {
    /// Layer construction failed
    LayerConstructionFailed(String),
    /// Invalid network architecture
    InvalidArchitecture(String),
    /// Parameter initialization failed
    ParameterInitializationFailed(String),
    /// Forward pass error
    ForwardPassError(String),
    /// Invalid input
    InvalidInput { expected: String, actual: String },
    /// Missing required layer
    MissingLayer(String),
    /// Layer compatibility error
    LayerCompatibilityError { layer1: String, layer2: String },
}

impl fmt::Display for NetworkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NetworkError::LayerConstructionFailed(msg) => {
                write!(f, "Layer construction failed: {}", msg)
            }
            NetworkError::InvalidArchitecture(msg) => {
                write!(f, "Invalid network architecture: {}", msg)
            }
            NetworkError::ParameterInitializationFailed(msg) => {
                write!(f, "Parameter initialization failed: {}", msg)
            }
            NetworkError::ForwardPassError(msg) => write!(f, "Forward pass error: {}", msg),
            NetworkError::InvalidInput { expected, actual } => {
                write!(f, "Invalid input: expected {}, got {}", expected, actual)
            }
            NetworkError::MissingLayer(layer) => write!(f, "Missing required layer: {}", layer),
            NetworkError::LayerCompatibilityError { layer1, layer2 } => {
                write!(f, "Layer compatibility error: {} and {}", layer1, layer2)
            }
        }
    }
}

impl std::error::Error for NetworkError {}

/// Data processing errors
#[derive(Debug, Clone)]
pub enum DataError {
    /// File not found
    FileNotFound(String),
    /// Invalid data format
    InvalidFormat(String),
    /// Data corruption detected
    DataCorruption(String),
    /// Insufficient data
    InsufficientData { required: usize, available: usize },
    /// Data preprocessing failed
    PreprocessingFailed(String),
    /// Batch creation failed
    BatchCreationFailed(String),
    /// Data loading timeout
    LoadingTimeout(String),
}

impl fmt::Display for DataError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataError::FileNotFound(path) => write!(f, "File not found: {}", path),
            DataError::InvalidFormat(msg) => write!(f, "Invalid data format: {}", msg),
            DataError::DataCorruption(msg) => write!(f, "Data corruption: {}", msg),
            DataError::InsufficientData {
                required,
                available,
            } => {
                write!(
                    f,
                    "Insufficient data: required {}, available {}",
                    required, available
                )
            }
            DataError::PreprocessingFailed(msg) => write!(f, "Preprocessing failed: {}", msg),
            DataError::BatchCreationFailed(msg) => write!(f, "Batch creation failed: {}", msg),
            DataError::LoadingTimeout(msg) => write!(f, "Data loading timeout: {}", msg),
        }
    }
}

impl std::error::Error for DataError {}

/// Memory allocation errors
#[derive(Debug, Clone)]
pub enum MemoryError {
    /// Out of memory
    OutOfMemory { requested: usize, available: usize },
    /// Memory allocation failed
    AllocationFailed(usize),
    /// Memory alignment error
    AlignmentError { required: usize, actual: usize },
    /// Memory corruption detected
    MemoryCorruption(String),
    /// Invalid memory access
    InvalidAccess(String),
}

impl fmt::Display for MemoryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemoryError::OutOfMemory {
                requested,
                available,
            } => {
                write!(
                    f,
                    "Out of memory: requested {} bytes, available {} bytes",
                    requested, available
                )
            }
            MemoryError::AllocationFailed(size) => {
                write!(f, "Memory allocation failed for {} bytes", size)
            }
            MemoryError::AlignmentError { required, actual } => {
                write!(
                    f,
                    "Memory alignment error: required {}, actual {}",
                    required, actual
                )
            }
            MemoryError::MemoryCorruption(msg) => write!(f, "Memory corruption: {}", msg),
            MemoryError::InvalidAccess(msg) => write!(f, "Invalid memory access: {}", msg),
        }
    }
}

impl std::error::Error for MemoryError {}

/// Configuration errors
#[derive(Debug, Clone)]
pub enum ConfigError {
    /// Invalid configuration value
    InvalidValue {
        key: String,
        value: String,
        reason: String,
    },
    /// Missing required configuration
    MissingRequired(String),
    /// Configuration parsing failed
    ParsingFailed(String),
    /// Validation failed
    ValidationFailed(String),
    /// Incompatible configuration
    IncompatibleConfiguration(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::InvalidValue { key, value, reason } => {
                write!(
                    f,
                    "Invalid configuration value for '{}': '{}' ({})",
                    key, value, reason
                )
            }
            ConfigError::MissingRequired(key) => {
                write!(f, "Missing required configuration: {}", key)
            }
            ConfigError::ParsingFailed(msg) => write!(f, "Configuration parsing failed: {}", msg),
            ConfigError::ValidationFailed(msg) => {
                write!(f, "Configuration validation failed: {}", msg)
            }
            ConfigError::IncompatibleConfiguration(msg) => {
                write!(f, "Incompatible configuration: {}", msg)
            }
        }
    }
}

impl std::error::Error for ConfigError {}

// Conversion implementations
impl From<TensorError> for MiniBurnError {
    fn from(error: TensorError) -> Self {
        MiniBurnError::Tensor(error)
    }
}

impl From<ShapeError> for MiniBurnError {
    fn from(error: ShapeError) -> Self {
        MiniBurnError::Shape(error)
    }
}

impl From<BackendError> for MiniBurnError {
    fn from(error: BackendError) -> Self {
        MiniBurnError::Backend(error)
    }
}

impl From<AutodiffError> for MiniBurnError {
    fn from(error: AutodiffError) -> Self {
        MiniBurnError::Autodiff(error)
    }
}

impl From<OptimizerError> for MiniBurnError {
    fn from(error: OptimizerError) -> Self {
        MiniBurnError::Optimizer(error)
    }
}

impl From<TrainingError> for MiniBurnError {
    fn from(error: TrainingError) -> Self {
        MiniBurnError::Training(error)
    }
}

impl From<crate::broadcast::BroadcastError> for MiniBurnError {
    fn from(error: crate::broadcast::BroadcastError) -> Self {
        MiniBurnError::Broadcasting(error)
    }
}

impl From<NetworkError> for MiniBurnError {
    fn from(error: NetworkError) -> Self {
        MiniBurnError::Network(error)
    }
}

impl From<DataError> for MiniBurnError {
    fn from(error: DataError) -> Self {
        MiniBurnError::Data(error)
    }
}

impl From<MemoryError> for MiniBurnError {
    fn from(error: MemoryError) -> Self {
        MiniBurnError::Memory(error)
    }
}

impl From<ConfigError> for MiniBurnError {
    fn from(error: ConfigError) -> Self {
        MiniBurnError::Config(error)
    }
}

impl From<std::io::Error> for MiniBurnError {
    fn from(error: std::io::Error) -> Self {
        MiniBurnError::Io(error.to_string())
    }
}

/// Result type alias for Mini-Burn operations
pub type Result<T> = std::result::Result<T, MiniBurnError>;

/// Trait for providing additional context to errors
pub trait ErrorContext<T> {
    /// Add context to an error
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;

    /// Add simple string context to an error
    fn context(self, msg: &str) -> Result<T>;
}

impl<T, E> ErrorContext<T> for std::result::Result<T, E>
where
    E: Into<MiniBurnError>,
{
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| {
            let context = f();
            let original_error = e.into();
            MiniBurnError::General(format!("{}: {}", context, original_error))
        })
    }

    fn context(self, msg: &str) -> Result<T> {
        self.with_context(|| msg.to_string())
    }
}

/// Macro for creating errors with context
#[macro_export]
macro_rules! mini_burn_error {
    ($kind:ident, $($arg:tt)*) => {
        $crate::error::MiniBurnError::$kind(format!($($arg)*))
    };
}

/// Macro for ensuring a condition or returning an error
#[macro_export]
macro_rules! ensure {
    ($cond:expr, $kind:ident, $($arg:tt)*) => {
        if !($cond) {
            return Err($crate::mini_burn_error!($kind, $($arg)*));
        }
    };
}

/// Macro for creating a bail macro (early return with error)
#[macro_export]
macro_rules! bail {
    ($kind:ident, $($arg:tt)*) => {
        return Err($crate::mini_burn_error!($kind, $($arg)*));
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[test]
    fn test_error_display() {
        let tensor_error = TensorError::ShapeMismatch {
            lhs: vec![2, 3],
            rhs: vec![3, 4],
            operation: "matmul".to_string(),
        };
        let error = MiniBurnError::Tensor(tensor_error);
        assert!(format!("{}", error).contains("Shape mismatch"));
        assert!(format!("{}", error).contains("matmul"));
    }

    #[test]
    fn test_error_conversion() {
        let tensor_error = TensorError::DivisionByZero;
        let mini_burn_error: MiniBurnError = tensor_error.into();
        matches!(
            mini_burn_error,
            MiniBurnError::Tensor(TensorError::DivisionByZero)
        );
    }

    #[test]
    fn test_error_context() {
        let result: std::result::Result<(), TensorError> = Err(TensorError::DivisionByZero);
        let with_context = result.context("During matrix multiplication");

        assert!(with_context.is_err());
        let error_msg = format!("{}", with_context.unwrap_err());
        assert!(error_msg.contains("During matrix multiplication"));
        assert!(error_msg.contains("Division by zero"));
    }

    #[test]
    fn test_error_macros() {
        let error = mini_burn_error!(General, "Test error: {}", 42);
        matches!(error, MiniBurnError::General(_));

        let error_msg = format!("{}", error);
        assert!(error_msg.contains("Test error: 42"));
    }

    #[test]
    fn test_ensure_macro() {
        fn test_function(x: i32) -> Result<i32> {
            ensure!(x > 0, General, "Value must be positive, got {}", x);
            Ok(x * 2)
        }

        assert!(test_function(5).is_ok());
        assert!(test_function(-1).is_err());

        let error = test_function(-1).unwrap_err();
        let error_msg = format!("{}", error);
        assert!(error_msg.contains("Value must be positive"));
    }

    #[test]
    fn test_error_source_chain() {
        let tensor_error = TensorError::InvalidDataType {
            expected: "f32".to_string(),
            actual: "i32".to_string(),
        };
        let mini_burn_error = MiniBurnError::Tensor(tensor_error);

        assert!(mini_burn_error.source().is_some());
    }

    #[test]
    fn test_memory_error() {
        let error = MemoryError::OutOfMemory {
            requested: 1024,
            available: 512,
        };
        let error_msg = format!("{}", error);
        assert!(error_msg.contains("1024"));
        assert!(error_msg.contains("512"));
    }

    #[test]
    fn test_config_error() {
        let error = ConfigError::InvalidValue {
            key: "learning_rate".to_string(),
            value: "-0.1".to_string(),
            reason: "must be positive".to_string(),
        };
        let error_msg = format!("{}", error);
        assert!(error_msg.contains("learning_rate"));
        assert!(error_msg.contains("-0.1"));
        assert!(error_msg.contains("must be positive"));
    }
}
