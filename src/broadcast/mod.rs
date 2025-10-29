//! Broadcasting Module
//!
//! This module implements NumPy-style broadcasting for automatic shape compatibility
//! in tensor operations, making element-wise operations more convenient and intuitive.

use crate::shape::Shape;
use std::cmp::max;

/// Error types for broadcasting operations
#[derive(Debug, Clone, PartialEq)]
pub enum BroadcastError {
    /// Shapes are incompatible for broadcasting
    IncompatibleShapes { lhs: Vec<usize>, rhs: Vec<usize> },
    /// Invalid dimension for broadcasting
    InvalidDimension { dim: usize, size: usize },
    /// General broadcasting error
    General(String),
}

impl std::fmt::Display for BroadcastError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BroadcastError::IncompatibleShapes { lhs, rhs } => {
                write!(
                    f,
                    "Incompatible shapes for broadcasting: {:?} and {:?}",
                    lhs, rhs
                )
            }
            BroadcastError::InvalidDimension { dim, size } => {
                write!(f, "Invalid dimension {} with size {}", dim, size)
            }
            BroadcastError::General(msg) => write!(f, "Broadcasting error: {}", msg),
        }
    }
}

impl std::error::Error for BroadcastError {}

/// Broadcasting rules and utilities
pub struct Broadcasting;

impl Broadcasting {
    /// Check if two shapes are compatible for broadcasting
    pub fn are_compatible(lhs: &[usize], rhs: &[usize]) -> bool {
        Self::broadcast_shape(lhs, rhs).is_ok()
    }

    /// Compute the broadcasted shape for two input shapes
    pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, BroadcastError> {
        let max_dims = max(lhs.len(), rhs.len());
        let mut result = Vec::with_capacity(max_dims);

        // Pad shorter shape with 1s at the beginning
        let lhs_padded = Self::pad_shape(lhs, max_dims);
        let rhs_padded = Self::pad_shape(rhs, max_dims);

        // Check compatibility and compute result shape
        for i in 0..max_dims {
            let lhs_dim = lhs_padded[i];
            let rhs_dim = rhs_padded[i];

            if lhs_dim == rhs_dim {
                result.push(lhs_dim);
            } else if lhs_dim == 1 {
                result.push(rhs_dim);
            } else if rhs_dim == 1 {
                result.push(lhs_dim);
            } else {
                return Err(BroadcastError::IncompatibleShapes {
                    lhs: lhs.to_vec(),
                    rhs: rhs.to_vec(),
                });
            }
        }

        Ok(result)
    }

    /// Pad a shape with 1s at the beginning to reach target dimensions
    fn pad_shape(shape: &[usize], target_dims: usize) -> Vec<usize> {
        let mut padded = vec![1; target_dims];
        let offset = target_dims - shape.len();
        padded[offset..].copy_from_slice(shape);
        padded
    }

    /// Get broadcasting strides for a shape to broadcast to a target shape
    pub fn broadcast_strides(
        original_shape: &[usize],
        target_shape: &[usize],
    ) -> Result<Vec<usize>, BroadcastError> {
        if !Self::are_compatible(original_shape, target_shape) {
            return Err(BroadcastError::IncompatibleShapes {
                lhs: original_shape.to_vec(),
                rhs: target_shape.to_vec(),
            });
        }

        let max_dims = target_shape.len();
        let mut strides = Vec::with_capacity(max_dims);
        let padded_original = Self::pad_shape(original_shape, max_dims);

        // Calculate strides for each dimension
        let mut stride = 1;
        for i in (0..max_dims).rev() {
            if padded_original[i] == 1 && target_shape[i] > 1 {
                // Broadcasting dimension - stride is 0
                strides.insert(0, 0);
            } else {
                // Normal dimension - use calculated stride
                strides.insert(0, stride);
                stride *= padded_original[i];
            }
        }

        Ok(strides)
    }

    /// Check if a shape can be broadcast to a target shape
    pub fn can_broadcast_to(shape: &[usize], target: &[usize]) -> bool {
        if shape.len() > target.len() {
            return false;
        }

        let offset = target.len() - shape.len();
        for (i, &dim) in shape.iter().enumerate() {
            let target_dim = target[offset + i];
            if dim != 1 && dim != target_dim {
                return false;
            }
        }

        true
    }

    /// Get the indices for broadcasting an array to a larger shape
    pub fn broadcast_indices(
        flat_index: usize,
        original_shape: &[usize],
        target_shape: &[usize],
    ) -> Result<usize, BroadcastError> {
        let strides = Self::broadcast_strides(original_shape, target_shape)?;
        let target_strides = Self::compute_strides(target_shape);

        let mut original_index = 0;
        let mut remaining = flat_index;

        for i in 0..target_shape.len() {
            let coord = remaining / target_strides[i];
            remaining %= target_strides[i];

            if strides[i] > 0 {
                original_index += coord * strides[i];
            }
        }

        Ok(original_index)
    }

    /// Compute strides for a given shape
    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let mut strides = Vec::with_capacity(shape.len());
        let mut stride = 1;

        for &dim in shape.iter().rev() {
            strides.insert(0, stride);
            stride *= dim;
        }

        strides
    }

    /// Reduce a shape by removing size-1 dimensions
    pub fn squeeze_shape(shape: &[usize]) -> Vec<usize> {
        shape.iter().copied().filter(|&dim| dim != 1).collect()
    }

    /// Add size-1 dimensions to expand a shape
    pub fn unsqueeze_shape(shape: &[usize], dim: usize) -> Result<Vec<usize>, BroadcastError> {
        if dim > shape.len() {
            return Err(BroadcastError::InvalidDimension {
                dim,
                size: shape.len(),
            });
        }

        let mut result = shape.to_vec();
        result.insert(dim, 1);
        Ok(result)
    }

    /// Get the shape after reducing along specific axes
    pub fn reduce_shape(shape: &[usize], axes: &[usize], keep_dims: bool) -> Vec<usize> {
        if axes.is_empty() {
            // Reduce all dimensions
            if keep_dims {
                vec![1; shape.len()]
            } else {
                vec![1]
            }
        } else {
            let mut result = shape.to_vec();
            for &axis in axes.iter().rev() {
                if axis < result.len() {
                    if keep_dims {
                        result[axis] = 1;
                    } else {
                        result.remove(axis);
                    }
                }
            }
            if result.is_empty() {
                vec![1]
            } else {
                result
            }
        }
    }
}

/// Trait for types that support broadcasting
pub trait Broadcastable {
    /// Get the shape of this broadcastable object
    fn shape(&self) -> &[usize];

    /// Check if this object can be broadcast with another
    fn can_broadcast_with<T: Broadcastable>(&self, other: &T) -> bool {
        Broadcasting::are_compatible(self.shape(), other.shape())
    }

    /// Get the broadcasted shape when combined with another broadcastable
    fn broadcast_shape_with<T: Broadcastable>(
        &self,
        other: &T,
    ) -> Result<Vec<usize>, BroadcastError> {
        Broadcasting::broadcast_shape(self.shape(), other.shape())
    }
}

/// Implementation of Broadcastable for slices (representing shapes)
impl Broadcastable for &[usize] {
    fn shape(&self) -> &[usize] {
        self
    }
}

impl Broadcastable for Vec<usize> {
    fn shape(&self) -> &[usize] {
        self.as_slice()
    }
}

impl<const N: usize> Broadcastable for Shape<N> {
    fn shape(&self) -> &[usize] {
        self.dims()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compatible_shapes() {
        // Same shapes
        assert!(Broadcasting::are_compatible(&[2, 3], &[2, 3]));

        // Broadcasting with 1s
        assert!(Broadcasting::are_compatible(&[1, 3], &[2, 3]));
        assert!(Broadcasting::are_compatible(&[2, 1], &[2, 3]));
        assert!(Broadcasting::are_compatible(&[1, 1], &[2, 3]));

        // Different dimensions
        assert!(Broadcasting::are_compatible(&[3], &[2, 3]));
        assert!(Broadcasting::are_compatible(&[2, 3], &[3]));

        // Incompatible shapes
        assert!(!Broadcasting::are_compatible(&[2, 3], &[2, 4]));
        assert!(!Broadcasting::are_compatible(&[3, 2], &[2, 3]));
    }

    #[test]
    fn test_broadcast_shape() {
        // Same shapes
        assert_eq!(
            Broadcasting::broadcast_shape(&[2, 3], &[2, 3]).unwrap(),
            vec![2, 3]
        );

        // Broadcasting with 1s
        assert_eq!(
            Broadcasting::broadcast_shape(&[1, 3], &[2, 3]).unwrap(),
            vec![2, 3]
        );
        assert_eq!(
            Broadcasting::broadcast_shape(&[2, 1], &[2, 3]).unwrap(),
            vec![2, 3]
        );

        // Different dimensions
        assert_eq!(
            Broadcasting::broadcast_shape(&[3], &[2, 3]).unwrap(),
            vec![2, 3]
        );
        assert_eq!(
            Broadcasting::broadcast_shape(&[2, 3], &[3]).unwrap(),
            vec![2, 3]
        );

        // Complex case
        assert_eq!(
            Broadcasting::broadcast_shape(&[1, 4, 1], &[3, 1, 5]).unwrap(),
            vec![3, 4, 5]
        );

        // Incompatible shapes should error
        assert!(Broadcasting::broadcast_shape(&[2, 3], &[2, 4]).is_err());
    }

    #[test]
    fn test_pad_shape() {
        assert_eq!(Broadcasting::pad_shape(&[3], 3), vec![1, 1, 3]);
        assert_eq!(Broadcasting::pad_shape(&[2, 3], 3), vec![1, 2, 3]);
        assert_eq!(Broadcasting::pad_shape(&[1, 2, 3], 3), vec![1, 2, 3]);
    }

    #[test]
    fn test_broadcast_strides() {
        // [3] -> [2, 3]
        let strides = Broadcasting::broadcast_strides(&[3], &[2, 3]).unwrap();
        assert_eq!(strides, vec![0, 1]);

        // [1, 3] -> [2, 3]
        let strides = Broadcasting::broadcast_strides(&[1, 3], &[2, 3]).unwrap();
        assert_eq!(strides, vec![0, 1]);

        // [2, 1] -> [2, 3]
        let strides = Broadcasting::broadcast_strides(&[2, 1], &[2, 3]).unwrap();
        assert_eq!(strides, vec![1, 0]);
    }

    #[test]
    fn test_can_broadcast_to() {
        assert!(Broadcasting::can_broadcast_to(&[3], &[2, 3]));
        assert!(Broadcasting::can_broadcast_to(&[1, 3], &[2, 3]));
        assert!(Broadcasting::can_broadcast_to(&[1], &[2, 3, 4]));

        assert!(!Broadcasting::can_broadcast_to(&[2, 3], &[3]));
        assert!(!Broadcasting::can_broadcast_to(&[2, 4], &[2, 3]));
    }

    #[test]
    fn test_squeeze_shape() {
        assert_eq!(Broadcasting::squeeze_shape(&[1, 3, 1, 4, 1]), vec![3, 4]);
        assert_eq!(Broadcasting::squeeze_shape(&[2, 3, 4]), vec![2, 3, 4]);
        assert_eq!(Broadcasting::squeeze_shape(&[1, 1, 1]), vec![]);
    }

    #[test]
    fn test_unsqueeze_shape() {
        assert_eq!(
            Broadcasting::unsqueeze_shape(&[2, 3], 0).unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(
            Broadcasting::unsqueeze_shape(&[2, 3], 1).unwrap(),
            vec![2, 1, 3]
        );
        assert_eq!(
            Broadcasting::unsqueeze_shape(&[2, 3], 2).unwrap(),
            vec![2, 3, 1]
        );

        // Invalid dimension
        assert!(Broadcasting::unsqueeze_shape(&[2, 3], 3).is_err());
    }

    #[test]
    fn test_reduce_shape() {
        // Reduce specific axes
        assert_eq!(
            Broadcasting::reduce_shape(&[2, 3, 4], &[1], false),
            vec![2, 4]
        );
        assert_eq!(
            Broadcasting::reduce_shape(&[2, 3, 4], &[1], true),
            vec![2, 1, 4]
        );

        // Reduce multiple axes
        assert_eq!(
            Broadcasting::reduce_shape(&[2, 3, 4], &[0, 2], false),
            vec![3]
        );
        assert_eq!(
            Broadcasting::reduce_shape(&[2, 3, 4], &[0, 2], true),
            vec![1, 3, 1]
        );

        // Reduce all axes
        assert_eq!(Broadcasting::reduce_shape(&[2, 3, 4], &[], false), vec![1]);
        assert_eq!(
            Broadcasting::reduce_shape(&[2, 3, 4], &[], true),
            vec![1, 1, 1]
        );
    }

    #[test]
    fn test_broadcastable_trait() {
        let shape1: &[usize] = &[2, 3];
        let shape2: &[usize] = &[1, 3];

        assert!(shape1.can_broadcast_with(&shape2));
        assert_eq!(shape1.broadcast_shape_with(&shape2).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_broadcast_indices() {
        // Simple case: [3] broadcasted to [2, 3]
        // Original indices: 0, 1, 2
        // Target indices: 0->0, 1->1, 2->2, 3->0, 4->1, 5->2
        assert_eq!(
            Broadcasting::broadcast_indices(0, &[3], &[2, 3]).unwrap(),
            0
        );
        assert_eq!(
            Broadcasting::broadcast_indices(1, &[3], &[2, 3]).unwrap(),
            1
        );
        assert_eq!(
            Broadcasting::broadcast_indices(3, &[3], &[2, 3]).unwrap(),
            0
        );
        assert_eq!(
            Broadcasting::broadcast_indices(4, &[3], &[2, 3]).unwrap(),
            1
        );
    }

    #[test]
    fn test_error_display() {
        let error = BroadcastError::IncompatibleShapes {
            lhs: vec![2, 3],
            rhs: vec![2, 4],
        };
        assert_eq!(
            format!("{}", error),
            "Incompatible shapes for broadcasting: [2, 3] and [2, 4]"
        );

        let error = BroadcastError::InvalidDimension { dim: 5, size: 3 };
        assert_eq!(format!("{}", error), "Invalid dimension 5 with size 3");
    }
}
