//! Data types supported by the tensor system

use crate::DefaultFloat;
use crate::DefaultInt;

/// Trait for data types that can be stored in tensors
pub trait DataType: Clone + Send + Sync + 'static {
    /// The primitive type that represents this data type
    type Primitive: Clone + Send + Sync + 'static;

    /// Name of the data type for debugging
    fn name() -> &'static str;
}

/// Float data type marker
#[derive(Clone, Debug)]
pub struct Float;

/// Integer data type marker
#[derive(Clone, Debug)]
pub struct Int;

/// Boolean data type marker
#[derive(Clone, Debug)]
pub struct Bool;

impl DataType for Float {
    type Primitive = DefaultFloat;

    fn name() -> &'static str {
        "Float"
    }
}

impl DataType for Int {
    type Primitive = DefaultInt;

    fn name() -> &'static str {
        "Int"
    }
}

impl DataType for Bool {
    type Primitive = bool;

    fn name() -> &'static str {
        "Bool"
    }
}

/// Element trait for individual tensor elements
pub trait Element: Clone + Send + Sync + 'static {
    /// Convert to float for calculations
    fn to_float(&self) -> DefaultFloat;

    /// Convert from float
    fn from_float(val: DefaultFloat) -> Self;
}

impl Element for DefaultFloat {
    fn to_float(&self) -> DefaultFloat {
        *self
    }

    fn from_float(val: DefaultFloat) -> Self {
        val
    }
}

impl Element for DefaultInt {
    fn to_float(&self) -> DefaultFloat {
        *self as DefaultFloat
    }

    fn from_float(val: DefaultFloat) -> Self {
        val as DefaultInt
    }
}

impl Element for bool {
    fn to_float(&self) -> DefaultFloat {
        if *self { 1.0 } else { 0.0 }
    }

    fn from_float(val: DefaultFloat) -> Self {
        val != 0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_type_names() {
        assert_eq!(Float::name(), "Float");
        assert_eq!(Int::name(), "Int");
        assert_eq!(Bool::name(), "Bool");
    }

    #[test]
    fn test_element_conversions() {
        // Float conversions
        let f = 3.14f32;
        assert_eq!(f.to_float(), 3.14);
        assert_eq!(DefaultFloat::from_float(2.71), 2.71);

        // Int conversions
        let i = 42i32;
        assert_eq!(i.to_float(), 42.0);
        assert_eq!(DefaultInt::from_float(3.14), 3);

        // Bool conversions
        assert_eq!(true.to_float(), 1.0);
        assert_eq!(false.to_float(), 0.0);
        assert_eq!(bool::from_float(1.0), true);
        assert_eq!(bool::from_float(0.0), false);
        assert_eq!(bool::from_float(0.5), true);
    }
}
