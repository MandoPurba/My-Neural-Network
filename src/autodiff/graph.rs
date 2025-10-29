//! Computation Graph for Automatic Differentiation
//!
//! This module implements a computation graph that tracks operations
//! and their dependencies for reverse-mode automatic differentiation.

use crate::backend::Backend;
use crate::data::DataType;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

/// Unique identifier for nodes in the computation graph
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(u64);

impl NodeId {
    /// Create a new unique node ID
    pub fn new() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Get the raw ID value
    pub fn id(&self) -> u64 {
        self.0
    }
}

impl Default for NodeId {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents an operation in the computation graph
#[derive(Debug, Clone)]
pub enum Operation {
    /// Leaf node (input tensor)
    Leaf,
    /// Addition operation
    Add { lhs: NodeId, rhs: NodeId },
    /// Subtraction operation
    Sub { lhs: NodeId, rhs: NodeId },
    /// Multiplication operation
    Mul { lhs: NodeId, rhs: NodeId },
    /// Division operation
    Div { lhs: NodeId, rhs: NodeId },
    /// Matrix multiplication
    MatMul { lhs: NodeId, rhs: NodeId },
    /// Scalar addition
    AddScalar { input: NodeId, scalar: f64 },
    /// Scalar multiplication
    MulScalar { input: NodeId, scalar: f64 },
    /// Exponential function
    Exp { input: NodeId },
    /// Natural logarithm
    Ln { input: NodeId },
    /// Power function
    Pow { input: NodeId, exponent: f64 },
    /// Square root
    Sqrt { input: NodeId },
    /// Sine function
    Sin { input: NodeId },
    /// Cosine function
    Cos { input: NodeId },
    /// Hyperbolic tangent
    Tanh { input: NodeId },
    /// Sigmoid function
    Sigmoid { input: NodeId },
    /// ReLU activation
    Relu { input: NodeId },
    /// Sum reduction
    Sum { input: NodeId },
    /// Mean reduction
    Mean { input: NodeId },
    /// Reshape operation
    Reshape { input: NodeId },
    /// Transpose operation
    Transpose { input: NodeId },
}

/// Information about a node in the computation graph
pub struct NodeInfo<B: Backend> {
    /// The operation that created this node
    pub operation: Operation,
    /// Shape information for gradient computation
    pub shape: Vec<usize>,
    /// Data type information
    pub dtype: String,
    /// Whether this node requires gradient computation
    pub requires_grad: bool,
    /// Dependencies (input nodes) for this operation
    pub dependencies: Vec<NodeId>,
    /// Reference count for memory management
    pub ref_count: usize,
    /// Gradient function for backward pass
    pub grad_fn: Option<Box<dyn GradFn<B> + Send + Sync>>,
}

impl<B: Backend> std::fmt::Debug for NodeInfo<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NodeInfo")
            .field("operation", &self.operation)
            .field("shape", &self.shape)
            .field("dtype", &self.dtype)
            .field("requires_grad", &self.requires_grad)
            .field("dependencies", &self.dependencies)
            .field("ref_count", &self.ref_count)
            .field("grad_fn", &self.grad_fn.is_some())
            .finish()
    }
}

/// Trait for gradient functions
pub trait GradFn<B: Backend> {
    /// Compute gradients for the inputs given the output gradient
    fn apply(&self, grad_output: &dyn std::any::Any) -> Vec<Box<dyn std::any::Any>>;

    /// Get the input node IDs this gradient function depends on
    fn input_ids(&self) -> Vec<NodeId>;
}

/// The computation graph that tracks all operations
#[derive(Debug)]
pub struct ComputationGraph<B: Backend> {
    /// Map from node IDs to their information
    nodes: HashMap<NodeId, NodeInfo<B>>,
    /// Root nodes (nodes with no dependencies)
    roots: Vec<NodeId>,
    /// Topological ordering for backward pass
    topo_order: Vec<NodeId>,
    /// Whether the graph needs recomputation of topological order
    needs_topo_update: bool,
}

impl<B: Backend> ComputationGraph<B> {
    /// Create a new empty computation graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            roots: Vec::new(),
            topo_order: Vec::new(),
            needs_topo_update: false,
        }
    }

    /// Add a leaf node (input tensor) to the graph
    pub fn add_leaf<const D: usize, K: DataType>(
        &mut self,
        node_id: NodeId,
        shape: &[usize],
        requires_grad: bool,
    ) -> NodeId {
        let node_info = NodeInfo {
            operation: Operation::Leaf,
            shape: shape.to_vec(),
            dtype: std::any::type_name::<K::Primitive>().to_string(),
            requires_grad,
            dependencies: Vec::new(),
            ref_count: 1,
            grad_fn: None,
        };

        self.nodes.insert(node_id, node_info);
        if requires_grad {
            self.roots.push(node_id);
        }
        self.needs_topo_update = true;

        node_id
    }

    /// Add an operation node to the graph
    pub fn add_operation(
        &mut self,
        operation: Operation,
        shape: Vec<usize>,
        dtype: String,
        requires_grad: bool,
        dependencies: Vec<NodeId>,
        grad_fn: Option<Box<dyn GradFn<B> + Send + Sync>>,
    ) -> NodeId {
        let node_id = NodeId::new();
        let node_info = NodeInfo {
            operation,
            shape,
            dtype,
            requires_grad,
            dependencies: dependencies.clone(),
            ref_count: 1,
            grad_fn,
        };

        self.nodes.insert(node_id, node_info);

        // Update reference counts for dependencies
        for dep_id in &dependencies {
            if let Some(dep_node) = self.nodes.get_mut(dep_id) {
                dep_node.ref_count += 1;
            }
        }

        self.needs_topo_update = true;
        node_id
    }

    /// Get information about a node
    pub fn get_node(&self, node_id: NodeId) -> Option<&NodeInfo<B>> {
        self.nodes.get(&node_id)
    }

    /// Get mutable information about a node
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut NodeInfo<B>> {
        self.nodes.get_mut(&node_id)
    }

    /// Remove a node from the graph
    pub fn remove_node(&mut self, node_id: NodeId) {
        if let Some(node_info) = self.nodes.remove(&node_id) {
            // Decrease reference counts for dependencies
            for dep_id in &node_info.dependencies {
                if let Some(dep_node) = self.nodes.get_mut(dep_id) {
                    dep_node.ref_count -= 1;
                    // Remove nodes with zero references
                    if dep_node.ref_count == 0 {
                        self.remove_node(*dep_id);
                    }
                }
            }
        }
        self.needs_topo_update = true;
    }

    /// Get all nodes that require gradients
    pub fn grad_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, info)| info.requires_grad)
            .map(|(id, _)| *id)
            .collect()
    }

    /// Compute topological ordering of the graph
    pub fn compute_topological_order(&mut self) -> Vec<NodeId> {
        if !self.needs_topo_update && !self.topo_order.is_empty() {
            return self.topo_order.clone();
        }

        let mut topo_order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut visiting = std::collections::HashSet::new();

        // Perform DFS for topological sort
        for &node_id in self.nodes.keys() {
            if !visited.contains(&node_id) {
                self.dfs_topo_sort(node_id, &mut visited, &mut visiting, &mut topo_order);
            }
        }

        topo_order.reverse(); // Reverse to get correct topological order
        self.topo_order = topo_order.clone();
        self.needs_topo_update = false;

        topo_order
    }

    /// Depth-first search for topological sorting
    fn dfs_topo_sort(
        &self,
        node_id: NodeId,
        visited: &mut std::collections::HashSet<NodeId>,
        visiting: &mut std::collections::HashSet<NodeId>,
        topo_order: &mut Vec<NodeId>,
    ) {
        if visiting.contains(&node_id) {
            // Cycle detected - should not happen in a well-formed computation graph
            panic!("Cycle detected in computation graph at node {:?}", node_id);
        }

        if visited.contains(&node_id) {
            return;
        }

        visiting.insert(node_id);

        if let Some(node_info) = self.nodes.get(&node_id) {
            for &dep_id in &node_info.dependencies {
                self.dfs_topo_sort(dep_id, visited, visiting, topo_order);
            }
        }

        visiting.remove(&node_id);
        visited.insert(node_id);
        topo_order.push(node_id);
    }

    /// Get the backward pass order (reverse topological order)
    pub fn backward_order(&mut self) -> Vec<NodeId> {
        let mut order = self.compute_topological_order();
        order.reverse();
        order
    }

    /// Clear the entire graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.roots.clear();
        self.topo_order.clear();
        self.needs_topo_update = false;
    }

    /// Get the number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the graph is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get all leaf nodes (inputs)
    pub fn leaf_nodes(&self) -> Vec<NodeId> {
        self.nodes
            .iter()
            .filter(|(_, info)| matches!(info.operation, Operation::Leaf))
            .map(|(id, _)| *id)
            .collect()
    }

    /// Prune nodes that are not reachable from any node requiring gradients
    pub fn prune_unreachable(&mut self) {
        let grad_nodes = self.grad_nodes();
        let mut reachable = std::collections::HashSet::new();

        // Mark all nodes reachable from gradient-requiring nodes
        for &grad_node in &grad_nodes {
            self.mark_reachable(grad_node, &mut reachable);
        }

        // Remove unreachable nodes
        let all_nodes: Vec<NodeId> = self.nodes.keys().cloned().collect();
        for node_id in all_nodes {
            if !reachable.contains(&node_id) {
                self.nodes.remove(&node_id);
            }
        }

        self.needs_topo_update = true;
    }

    /// Mark all nodes reachable from a given node
    fn mark_reachable(&self, node_id: NodeId, reachable: &mut std::collections::HashSet<NodeId>) {
        if reachable.contains(&node_id) {
            return;
        }

        reachable.insert(node_id);

        if let Some(node_info) = self.nodes.get(&node_id) {
            for &dep_id in &node_info.dependencies {
                self.mark_reachable(dep_id, reachable);
            }
        }
    }
}

impl<B: Backend> Default for ComputationGraph<B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CpuBackend;

    #[test]
    fn test_node_id_generation() {
        let id1 = NodeId::new();
        let id2 = NodeId::new();
        assert_ne!(id1, id2);
        assert!(id2.id() > id1.id());
    }

    #[test]
    fn test_computation_graph_creation() {
        let mut graph = ComputationGraph::<CpuBackend>::new();
        assert!(graph.is_empty());
        assert_eq!(graph.node_count(), 0);
    }

    #[test]
    fn test_add_leaf_node() {
        let mut graph = ComputationGraph::<CpuBackend>::new();
        let node_id = NodeId::new();

        graph.add_leaf::<2, crate::Float>(node_id, &[2, 3], true);

        assert_eq!(graph.node_count(), 1);
        let node_info = graph.get_node(node_id).unwrap();
        assert!(matches!(node_info.operation, Operation::Leaf));
        assert_eq!(node_info.shape, vec![2, 3]);
        assert!(node_info.requires_grad);
    }

    #[test]
    fn test_add_operation_node() {
        let mut graph = ComputationGraph::<CpuBackend>::new();
        let input1 = NodeId::new();
        let input2 = NodeId::new();

        graph.add_leaf::<2, crate::Float>(input1, &[2, 3], true);
        graph.add_leaf::<2, crate::Float>(input2, &[2, 3], true);

        let add_node = graph.add_operation(
            Operation::Add {
                lhs: input1,
                rhs: input2,
            },
            vec![2, 3],
            "f32".to_string(),
            true,
            vec![input1, input2],
            None,
        );

        assert_eq!(graph.node_count(), 3);
        let node_info = graph.get_node(add_node).unwrap();
        assert!(matches!(node_info.operation, Operation::Add { .. }));
        assert_eq!(node_info.dependencies, vec![input1, input2]);
    }

    #[test]
    fn test_topological_ordering() {
        let mut graph = ComputationGraph::<CpuBackend>::new();
        let input1 = NodeId::new();
        let input2 = NodeId::new();

        graph.add_leaf::<2, crate::Float>(input1, &[2, 3], true);
        graph.add_leaf::<2, crate::Float>(input2, &[2, 3], true);

        let add_node = graph.add_operation(
            Operation::Add {
                lhs: input1,
                rhs: input2,
            },
            vec![2, 3],
            "f32".to_string(),
            true,
            vec![input1, input2],
            None,
        );

        let topo_order = graph.compute_topological_order();
        assert_eq!(topo_order.len(), 3);

        // The add operation should come after its inputs
        let add_pos = topo_order.iter().position(|&id| id == add_node).unwrap();
        let input1_pos = topo_order.iter().position(|&id| id == input1).unwrap();
        let input2_pos = topo_order.iter().position(|&id| id == input2).unwrap();

        assert!(add_pos > input1_pos);
        assert!(add_pos > input2_pos);
    }

    #[test]
    fn test_leaf_nodes() {
        let mut graph = ComputationGraph::<CpuBackend>::new();
        let leaf1 = NodeId::new();
        let leaf2 = NodeId::new();

        graph.add_leaf::<2, crate::Float>(leaf1, &[2, 3], true);
        graph.add_leaf::<2, crate::Float>(leaf2, &[2, 3], false);

        let _op_node = graph.add_operation(
            Operation::Add {
                lhs: leaf1,
                rhs: leaf2,
            },
            vec![2, 3],
            "f32".to_string(),
            true,
            vec![leaf1, leaf2],
            None,
        );

        let leaf_nodes = graph.leaf_nodes();
        assert_eq!(leaf_nodes.len(), 2);
        assert!(leaf_nodes.contains(&leaf1));
        assert!(leaf_nodes.contains(&leaf2));
    }

    #[test]
    fn test_grad_nodes() {
        let mut graph = ComputationGraph::<CpuBackend>::new();
        let grad_node = NodeId::new();
        let no_grad_node = NodeId::new();

        graph.add_leaf::<2, crate::Float>(grad_node, &[2, 3], true);
        graph.add_leaf::<2, crate::Float>(no_grad_node, &[2, 3], false);

        let grad_nodes = graph.grad_nodes();
        assert_eq!(grad_nodes.len(), 1);
        assert!(grad_nodes.contains(&grad_node));
        assert!(!grad_nodes.contains(&no_grad_node));
    }
}
