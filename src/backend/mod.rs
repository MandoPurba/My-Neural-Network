//! Backend abstraction for different hardware implementations

use crate::data::DataType;
use std::marker::PhantomData;

/// Main backend trait that defines operations available for tensor computation
pub trait Backend: Clone + Send + Sync + 'static {
    /// The device type this backend uses
    type Device: Clone + Send + Sync;

    /// Storage type for tensors
    type Storage<T: DataType>: Clone + Send + Sync;

    /// Get the default device for this backend
    fn default_device() -> Self::Device;

    /// Create storage from raw data
    fn from_data<T: DataType>(data: Vec<T::Primitive>, device: &Self::Device) -> Self::Storage<T>;

    /// Convert storage back to raw data
    fn to_data<T: DataType>(storage: &Self::Storage<T>) -> Vec<T::Primitive>;

    /// Get the size of storage
    fn storage_size<T: DataType>(storage: &Self::Storage<T>) -> usize;
}

/// CPU backend implementation
#[derive(Clone, Debug)]
pub struct CpuBackend;

/// CPU device (essentially a marker since CPU is always available)
#[derive(Clone, Debug)]
pub struct CpuDevice;

/// CPU storage that holds data in memory
#[derive(Clone, Debug)]
pub struct CpuStorage<T: DataType> {
    data: Vec<T::Primitive>,
    _phantom: PhantomData<T>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    type Device = CpuDevice;
    type Storage<T: DataType> = CpuStorage<T>;

    fn default_device() -> Self::Device {
        CpuDevice
    }

    fn from_data<T: DataType>(data: Vec<T::Primitive>, _device: &Self::Device) -> Self::Storage<T> {
        CpuStorage {
            data,
            _phantom: PhantomData,
        }
    }

    fn to_data<T: DataType>(storage: &Self::Storage<T>) -> Vec<T::Primitive> {
        storage.data.clone()
    }

    fn storage_size<T: DataType>(storage: &Self::Storage<T>) -> usize {
        storage.data.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Float;

    #[test]
    fn test_cpu_backend_creation() {
        let _backend = CpuBackend::new();
        let device = CpuBackend::default_device();

        let data = vec![1.0, 2.0, 3.0, 4.0];
        let storage = CpuBackend::from_data::<Float>(data.clone(), &device);

        assert_eq!(CpuBackend::to_data(&storage), data);
        assert_eq!(CpuBackend::storage_size(&storage), 4);
    }
}
