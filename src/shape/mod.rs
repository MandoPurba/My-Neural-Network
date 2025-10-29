//! Shape system for multi-dimensional tensor tracking

use std::fmt;

/// Shape represents the dimensions of a tensor
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape<const D: usize> {
    dims: [usize; D],
}

impl<const D: usize> Shape<D> {
    /// Create a new shape from dimensions
    pub fn new(dims: [usize; D]) -> Self {
        Self { dims }
    }

    /// Get the dimensions as a slice
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        self.dims.iter().product()
    }

    /// Get the total number of elements (alias for numel)
    pub fn num_elements(&self) -> usize {
        self.numel()
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        D
    }

    /// Get a specific dimension
    pub fn dim(&self, index: usize) -> usize {
        self.dims[index]
    }
}

impl<const D: usize> fmt::Display for Shape<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, dim) in self.dims.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", dim)?;
        }
        write!(f, "]")
    }
}

// Type aliases for common shapes
pub type Shape1D = Shape<1>;
pub type Shape2D = Shape<2>;
pub type Shape3D = Shape<3>;
pub type Shape4D = Shape<4>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_creation() {
        let shape = Shape::new([2, 3, 4]);
        assert_eq!(shape.dims(), &[2, 3, 4]);
        assert_eq!(shape.numel(), 24);
        assert_eq!(shape.ndim(), 3);
    }

    #[test]
    fn test_shape_indexing() {
        let shape = Shape::new([5, 10]);
        assert_eq!(shape.dim(0), 5);
        assert_eq!(shape.dim(1), 10);
    }

    #[test]
    fn test_shape_display() {
        let shape = Shape::new([2, 3, 4]);
        assert_eq!(format!("{}", shape), "[2, 3, 4]");
    }

    #[test]
    fn test_shape_equality() {
        let shape1 = Shape::new([2, 3]);
        let shape2 = Shape::new([2, 3]);
        let shape3 = Shape::new([3, 2]);

        assert_eq!(shape1, shape2);
        assert_ne!(shape1, shape3);
    }

    #[test]
    fn test_different_dimension_shapes() {
        let shape_1d = Shape::new([5]);
        let shape_2d = Shape::new([2, 3]);
        let shape_3d = Shape::new([2, 3, 4]);
        let shape_4d = Shape::new([2, 3, 4, 5]);

        assert_eq!(shape_1d.ndim(), 1);
        assert_eq!(shape_2d.ndim(), 2);
        assert_eq!(shape_3d.ndim(), 3);
        assert_eq!(shape_4d.ndim(), 4);

        assert_eq!(shape_1d.numel(), 5);
        assert_eq!(shape_2d.numel(), 6);
        assert_eq!(shape_3d.numel(), 24);
        assert_eq!(shape_4d.numel(), 120);
    }
}
