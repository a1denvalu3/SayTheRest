use libloading::Library;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TorchFlavor {
    Cpu,
    Cuda126,
    Cuda128,
    CudaLibraries,
}

impl TorchFlavor {
    pub fn extra(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda126 => "cu126",
            Self::Cuda128 => "cu128",
            Self::CudaLibraries => "onnx-cu12",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CudaDevice {
    pub driver_version: i32,
    pub count: i32,
}

impl CudaDevice {
    pub fn detect() -> Option<Self> {
        if std::env::var("CUDA_VISIBLE_DEVICES")
            .is_ok_and(|value| value.trim().is_empty() || value.trim() == "-1")
        {
            return None;
        }
        // Query the installed driver, not the presence of a vendor utility or
        // a PCI device. Loading the driver never downloads or installs software.
        let name = if cfg!(windows) {
            "nvcuda.dll"
        } else {
            "libcuda.so.1"
        };
        unsafe {
            let library = Library::new(name).ok()?;
            let init = library
                .get::<unsafe extern "system" fn(u32) -> i32>(b"cuInit\0")
                .ok()?;
            let count = library
                .get::<unsafe extern "system" fn(*mut i32) -> i32>(b"cuDeviceGetCount\0")
                .ok()?;
            let version = library
                .get::<unsafe extern "system" fn(*mut i32) -> i32>(b"cuDriverGetVersion\0")
                .ok()?;
            let mut device = Self {
                driver_version: 0,
                count: 0,
            };
            if init(0) != 0
                || count(&mut device.count) != 0
                || version(&mut device.driver_version) != 0
                || device.count <= 0
            {
                return None;
            }
            Some(device)
        }
    }

    pub fn torch_flavor(self) -> Option<TorchFlavor> {
        match self.driver_version {
            12080.. => Some(TorchFlavor::Cuda128),
            12060.. => Some(TorchFlavor::Cuda126),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuda_runtime_matches_driver_support() {
        assert_eq!(
            CudaDevice {
                driver_version: 12040,
                count: 1
            }
            .torch_flavor(),
            None
        );
        assert_eq!(
            CudaDevice {
                driver_version: 12060,
                count: 1
            }
            .torch_flavor(),
            Some(TorchFlavor::Cuda126)
        );
        assert_eq!(
            CudaDevice {
                driver_version: 12080,
                count: 1
            }
            .torch_flavor(),
            Some(TorchFlavor::Cuda128)
        );
        assert_eq!(
            CudaDevice {
                driver_version: 13000,
                count: 2
            }
            .torch_flavor(),
            Some(TorchFlavor::Cuda128)
        );
    }
}
