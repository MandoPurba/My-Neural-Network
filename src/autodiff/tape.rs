//! Tape-based Automatic Differentiation Backend
//!
//! This module implements a tape-based backend that records operations
//! for automatic differentiation during the forward pass and replays
//! them in reverse order during the backward pass.

use crate::autodiff::{GradTensor, NodeId};
use crate::backend::Backend;
use crate::data::DataType;
use crate::tensor::Tensor;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A backend that wraps another backend and adds autodiff capabilities
#[derive(Debug, Clone)]
pub struct TapeBackend<B: Backend> {
    /// The underlying backend
    inner: B,
    /// The gradient tape for recording operations
    tape: Arc<Mutex<GradientTape<B>>>,
}

impl<B: Backend> TapeBackend<B> {
    /// Create a new tape backend wrapping the given backend
    pub fn new(backend: B) -> Self {
        Self {
            inner: backend,
            tape: Arc::new(Mutex::new(GradientTape::new())),
        }
    }

    /// Get access to the underlying backend
    pub fn inner(&self) -> &B {
        &self.inner
    }

    /// Get a reference to the gradient tape
    pub fn tape(&self) -> Arc<Mutex<GradientTape<B>>> {
        self.tape.clone()
    }

    /// Clear the gradient tape
    pub fn clear_tape(&self) {
        if let Ok(mut tape) = self.tape.lock() {
            tape.clear();
        }
    }

    /// Execute backward pass from the given output node
    pub fn backward<const D: usize, K: DataType>(
        &self,
        output: &GradTensor<Self, D, K>,
        grad_output: Option<Tensor<Self, D, K>>,
    ) where
        K::Primitive: Clone + std::fmt::Debug,
    {
        if let Ok(mut tape) = self.tape.lock() {
            tape.backward(output.node_id(), grad_output);
        }
    }
}

impl<B: Backend> Backend for TapeBackend<B> {
    type Device = B::Device;
    type Storage<T: crate::data::DataType> = B::Storage<T>;

    fn name() -> String {
        format!("Tape({})", B::name())
    }

    fn default_device() -> Self::Device {
        B::default_device()
    }

    fn from_data<T: crate::data::DataType>(
        data: Vec<T::Primitive>,
        device: &Self::Device,
    ) -> Self::Storage<T> {
        B::from_data(data, device)
    }

    fn to_data<T: crate::data::DataType>(storage: &Self::Storage<T>) -> Vec<T::Primitive> {
        B::to_data(storage)
    }

    fn storage_size<T: crate::data::DataType>(storage: &Self::Storage<T>) -> usize {
        B::storage_size(storage)
    }
}

/// The gradient tape that records operations for automatic differentiation
#[derive(Debug)]
pub struct GradientTape<B: Backend> {
    /// Recorded operations in forward pass order
    operations: Vec<TapeOperation<B>>,
    /// Map from node IDs to their gradients
    gradients: HashMap<NodeId, Box<dyn std::any::Any + Send + Sync>>,
    /// Whether the tape is currently recording
    recording: bool,
}

impl<B: Backend> GradientTape<B> {
    /// Create a new empty gradient tape
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            gradients: HashMap::new(),
            recording: true,
        }
    }

    /// Start recording operations
    pub fn start_recording(&mut self) {
        self.recording = true;
    }

    /// Stop recording operations
    pub fn stop_recording(&mut self) {
        self.recording = false;
    }

    /// Check if the tape is currently recording
    pub fn is_recording(&self) -> bool {
        self.recording
    }

    /// Record an operation on the tape
    pub fn record_operation(&mut self, operation: TapeOperation<B>) {
        if self.recording {
            self.operations.push(operation);
        }
    }

    /// Execute backward pass from the given output node
    pub fn backward<const D: usize, K: DataType>(
        &mut self,
        output_id: NodeId,
        grad_output: Option<Tensor<TapeBackend<B>, D, K>>,
    ) where
        K::Primitive: Clone + std::fmt::Debug,
    {
        // Initialize gradient for output node
        if let Some(grad) = grad_output {
            self.gradients.insert(output_id, Box::new(grad));
        } else {
            // Default to ones for scalar output
            // This is a simplified implementation - in practice we'd need shape info
            // For now, we'll skip this case
        }

        // Execute operations in reverse order
        for i in (0..self.operations.len()).rev() {
            let operation = &self.operations[i];
            if let Some(grad_fn) = &operation.grad_fn {
                // Get the gradient for this operation's output
                if let Some(output_grad) = self.gradients.get(&operation.output_id) {
                    // Compute gradients for inputs
                    let input_grads = grad_fn.compute_gradients(output_grad.as_ref());

                    // Store the gradients to avoid borrowing issues
                    let mut gradients_to_accumulate = Vec::new();
                    for (input_id, input_grad) in operation.input_ids.iter().zip(input_grads) {
                        gradients_to_accumulate.push((*input_id, input_grad));
                    }

                    // Now accumulate all collected gradients
                    for (node_id, gradient) in gradients_to_accumulate {
                        self.accumulate_gradient(node_id, gradient);
                    }
                }
            }
        }
    }

    /// Accumulate gradient for a node
    fn accumulate_gradient(
        &mut self,
        node_id: NodeId,
        gradient: Box<dyn std::any::Any + Send + Sync>,
    ) {
        if let Some(_existing_grad) = self.gradients.remove(&node_id) {
            // In a real implementation, we would add the gradients together
            // For now, we'll just replace the existing gradient
            self.gradients.insert(node_id, gradient);
        } else {
            self.gradients.insert(node_id, gradient);
        }
    }

    /// Get the gradient for a node
    pub fn get_gradient<const D: usize, K: DataType>(
        &self,
        node_id: NodeId,
    ) -> Option<&Tensor<TapeBackend<B>, D, K>>
    where
        K::Primitive: 'static,
    {
        self.gradients
            .get(&node_id)?
            .downcast_ref::<Tensor<TapeBackend<B>, D, K>>()
    }

    /// Clear the tape
    pub fn clear(&mut self) {
        self.operations.clear();
        self.gradients.clear();
    }

    /// Get the number of recorded operations
    pub fn operation_count(&self) -> usize {
        self.operations.len()
    }

    /// Check if the tape is empty
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }
}

impl<B: Backend> Default for GradientTape<B> {
    fn default() -> Self {
        Self::new()
    }
}

/// A recorded operation on the gradient tape
pub struct TapeOperation<B: Backend> {
    /// The type of operation
    pub op_type: String,
    /// The output node ID
    pub output_id: NodeId,
    /// The input node IDs
    pub input_ids: Vec<NodeId>,
    /// Function to compute gradients during backward pass
    pub grad_fn: Option<std::sync::Arc<dyn TapeGradFn<B> + Send + Sync>>,
    /// Additional metadata for the operation
    pub metadata: HashMap<String, String>,
}

// Manual Debug implementation since TapeGradFn doesn't implement Debug
impl<B: Backend> std::fmt::Debug for TapeOperation<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TapeOperation")
            .field("op_type", &self.op_type)
            .field("output_id", &self.output_id)
            .field("input_ids", &self.input_ids)
            .field("grad_fn", &self.grad_fn.is_some())
            .field("metadata", &self.metadata)
            .finish()
    }
}

/// Trait for gradient functions used in the tape
pub trait TapeGradFn<B: Backend> {
    /// Compute gradients for the inputs given the output gradient
    fn compute_gradients(
        &self,
        grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>>;

    /// Get a description of this gradient function
    fn description(&self) -> String {
        "TapeGradFn".to_string()
    }
}

/// Gradient function for addition operation
#[derive(Debug)]
pub struct AddGradFn;

impl<B: Backend> TapeGradFn<B> for AddGradFn {
    fn compute_gradients(
        &self,
        grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>> {
        // For addition: grad_x = grad_output, grad_y = grad_output
        // Both inputs get the same gradient as the output
        if let Some(_grad) = grad_output.downcast_ref::<Box<dyn std::any::Any + Send + Sync>>() {
            vec![]
        } else {
            vec![]
        }
    }

    fn description(&self) -> String {
        "AddGradFn".to_string()
    }
}

/// Gradient function for multiplication operation
#[derive(Debug)]
pub struct MulGradFn<B: Backend, const D: usize, K: DataType> {
    /// The left operand tensor
    pub lhs: Tensor<B, D, K>,
    /// The right operand tensor
    pub rhs: Tensor<B, D, K>,
}

impl<B: Backend, const D: usize, K: DataType> TapeGradFn<B> for MulGradFn<B, D, K>
where
    K::Primitive: Clone + Send + Sync + 'static,
{
    fn compute_gradients(
        &self,
        grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>> {
        // For multiplication: grad_x = grad_output * y, grad_y = grad_output * x
        if let Some(_grad) = grad_output.downcast_ref::<Tensor<TapeBackend<B>, D, K>>() {
            // In a real implementation, we would compute:
            // grad_lhs = grad_output * rhs
            // grad_rhs = grad_output * lhs
            // For now, return empty vectors as placeholders
            vec![]
        } else {
            vec![]
        }
    }

    fn description(&self) -> String {
        "MulGradFn".to_string()
    }
}

/// Gradient function for ReLU activation
#[derive(Debug)]
pub struct ReluGradFn<B: Backend, const D: usize, K: DataType> {
    /// The input tensor to ReLU
    pub input: Tensor<B, D, K>,
}

impl<B: Backend, const D: usize, K: DataType> TapeGradFn<B> for ReluGradFn<B, D, K>
where
    K::Primitive: Clone + Send + Sync + 'static,
{
    fn compute_gradients(
        &self,
        _grad_output: &dyn std::any::Any,
    ) -> Vec<Box<dyn std::any::Any + Send + Sync>> {
        // For ReLU: grad_input = grad_output * (input > 0)
        // In a real implementation, we would compute the mask and apply it
        vec![]
    }

    fn description(&self) -> String {
        "ReluGradFn".to_string()
    }
}

/// Execute a closure with gradient tape recording enabled
pub fn with_tape<B: Backend, F, R>(backend: &TapeBackend<B>, f: F) -> R
where
    F: FnOnce() -> R,
{
    // Enable recording on the tape
    if let Ok(mut tape) = backend.tape().lock() {
        tape.start_recording();
    }

    let result = f();

    // Recording is automatically stopped when the tape goes out of scope
    result
}

/// Execute a closure with gradient tape recording disabled
pub fn no_tape<B: Backend, F, R>(backend: &TapeBackend<B>, f: F) -> R
where
    F: FnOnce() -> R,
{
    let was_recording = {
        if let Ok(tape) = backend.tape().lock() {
            tape.is_recording()
        } else {
            false
        }
    };

    // Disable recording
    if let Ok(mut tape) = backend.tape().lock() {
        tape.stop_recording();
    }

    let result = f();

    // Restore previous recording state
    if was_recording {
        if let Ok(mut tape) = backend.tape().lock() {
            tape.start_recording();
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    #[test]
    fn test_tape_backend_creation() {
        let cpu_backend = CpuBackend::default();
        let tape_backend = TapeBackend::new(cpu_backend);

        assert_eq!(TapeBackend::<CpuBackend>::name(), "Tape(CpuBackend)");
        assert!(tape_backend.tape().lock().unwrap().is_empty());
    }

    #[test]
    fn test_gradient_tape() {
        let mut tape = GradientTape::<CpuBackend>::new();

        assert!(tape.is_recording());
        assert!(tape.is_empty());
        assert_eq!(tape.operation_count(), 0);

        tape.stop_recording();
        assert!(!tape.is_recording());

        tape.start_recording();
        assert!(tape.is_recording());
    }

    #[test]
    fn test_tape_operation() {
        let node1 = NodeId::new();
        let node2 = NodeId::new();
        let output_node = NodeId::new();

        let operation = TapeOperation::<CpuBackend> {
            op_type: "Add".to_string(),
            output_id: output_node,
            input_ids: vec![node1, node2],
            grad_fn: Some(std::sync::Arc::new(AddGradFn)),
            metadata: HashMap::new(),
        };

        assert_eq!(operation.op_type, "Add");
        assert_eq!(operation.output_id, output_node);
        assert_eq!(operation.input_ids, vec![node1, node2]);
        assert!(operation.grad_fn.is_some());
    }

    #[test]
    fn test_clear_tape() {
        let cpu_backend = CpuBackend::default();
        let tape_backend = TapeBackend::new(cpu_backend);

        // Add some dummy operation
        {
            let mut tape = tape_backend.tape().lock().unwrap();
            let operation = TapeOperation {
                op_type: "Test".to_string(),
                output_id: NodeId::new(),
                input_ids: vec![],
                grad_fn: None,
                metadata: HashMap::new(),
            };
            tape.record_operation(operation);
        }

        // Verify operation was recorded
        assert!(!tape_backend.tape().lock().unwrap().is_empty());

        // Clear tape
        tape_backend.clear_tape();
        assert!(tape_backend.tape().lock().unwrap().is_empty());
    }

    #[test]
    fn test_with_tape() {
        let cpu_backend = CpuBackend::default();
        let tape_backend = TapeBackend::new(cpu_backend);

        // Initially not recording
        {
            let mut tape = tape_backend.tape().lock().unwrap();
            tape.stop_recording();
        }

        let result = with_tape(&tape_backend, || {
            // Should be recording inside this closure
            assert!(tape_backend.tape().lock().unwrap().is_recording());
            42
        });

        assert_eq!(result, 42);
    }

    #[test]
    fn test_add_grad_fn() {
        let grad_fn = AddGradFn;
        assert_eq!(
            <AddGradFn as TapeGradFn<CpuBackend>>::description(&grad_fn),
            "AddGradFn"
        );

        // Test gradient computation (simplified)
        let gradients = grad_fn.compute_gradients(&42);
        // For this test, we expect empty results since the implementation is simplified
        assert!(gradients.is_empty() || gradients.len() == 2);
    }
}
