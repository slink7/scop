

pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name
];

use std::collections::HashSet;

use log::*;

use anyhow::{anyhow, Result};

use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{KhrSurfaceExtensionInstanceCommands, StringArray};

use thiserror::Error;



#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);



#[derive(Clone, Debug, Default)]
pub struct SwapchainSupport {
    pub capabilities: vk::SurfaceCapabilitiesKHR,
    pub formats: Vec<vk::SurfaceFormatKHR>,
    pub present_modes: Vec<vk::PresentModeKHR>
}

impl SwapchainSupport {
    pub unsafe fn get(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice
    ) -> anyhow::Result<Self> {
        
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(
                    physical_device,
                    surface
                )?,
            formats: instance
                .get_physical_device_surface_formats_khr(
                    physical_device,
                    surface
                )?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(
                    physical_device,
                    surface
                )?
        })
    }
}



#[derive(Copy, Clone, Debug, Default)]
pub struct QueueFamilyIndices {
    pub graphics: u32,
    pub present: u32
}

impl QueueFamilyIndices {
    pub unsafe fn get(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        physical_device: vk::PhysicalDevice,
        queue_families: &Vec<vk::QueueFamilyProperties>
    ) -> anyhow::Result<Self> {
        
        let graphics = queue_families
            .iter()
            .position(|p| 
                p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, properties) in queue_families.iter().enumerate() {
            if instance.get_physical_device_surface_support_khr(
                physical_device,
                index as u32,
                surface
            )? {
                present = Some(index as u32);
                break ;
            }
        }

        if let (Some(graphics), Some(present)) = (graphics, present) {
            Ok(Self { graphics, present })
        } else {
            Err(anyhow!(SuitabilityError("Missing required queue families.")))
        }
    }
}



pub unsafe fn get_max_msaa_samples(
    properties: vk::PhysicalDeviceProperties
) -> vk::SampleCountFlags {
    let counts =
        properties.limits.framebuffer_color_sample_counts &
        properties.limits.framebuffer_depth_sample_counts;

    [
        vk::SampleCountFlags::_64,
        vk::SampleCountFlags::_32,
        vk::SampleCountFlags::_16,
        vk::SampleCountFlags::_8,
        vk::SampleCountFlags::_4,
        vk::SampleCountFlags::_2,
    ]
    .iter()
    .cloned()
    .find(|c| counts.contains(*c))
    .unwrap_or(vk::SampleCountFlags::_1)
}



#[derive(Clone, Debug, Default)]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    pub queue_families_indices: QueueFamilyIndices,
    pub swapchain_support: SwapchainSupport,
    pub extensions: HashSet<StringArray<256>>,
    pub max_msaa_samples: vk::SampleCountFlags
}

impl PhysicalDevice {

    unsafe fn from_handle(
        instance: &Instance,
        surface: vk::SurfaceKHR,
        handle: vk::PhysicalDevice
    ) -> Result<Self> {

        let properties = instance
            .get_physical_device_properties(handle);
        let features = instance
            .get_physical_device_features(handle);
        let memory_properties = instance
            .get_physical_device_memory_properties(handle);
        let queue_families = instance
            .get_physical_device_queue_family_properties(handle);
        let queue_families_indices = 
            QueueFamilyIndices::get(instance, surface, handle, &queue_families)?;
        let swapchain_support = 
            SwapchainSupport::get(instance, surface, handle)?;
        let extensions = instance
            .enumerate_device_extension_properties(handle, None)?
            .iter()
            .map(|e| e.extension_name)
            .collect::<HashSet<_>>();
        let max_msaa_samples = get_max_msaa_samples(properties);

        Ok(Self {
            handle,
            properties,
            features,
            memory_properties,
            queue_families,
            queue_families_indices,
            swapchain_support,
            extensions,
            max_msaa_samples
        })
    }

    pub unsafe fn new(
        instance: &vulkanalia::Instance,
        surface: vk::SurfaceKHR
    ) -> Result<Self> {
        
        for handle in instance.enumerate_physical_devices()? {
            let device = Self::from_handle(instance, surface, handle)?;
            
            if let Err(error) = device.is_suitable() {
                warn!(
                    "Skipping physical device (`{}`). {}",
                    device.properties.device_name,
                    error
                );
                continue ;
            }

            return Ok(device);
        }

        Err(anyhow!("Found no suitable physical device."))
    }
    
    pub unsafe fn is_suitable(&self) -> Result<()> {

        if self.properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
            return Err(anyhow!(SuitabilityError("Only discrete GPUs are supported.")));
        }
        if self.features.sampler_anisotropy != vk::TRUE {
            return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
        }
        if self.features.geometry_shader != vk::TRUE {
            return Err(anyhow!(SuitabilityError("Missing geometry shader support.")));
        }
        if self.swapchain_support.formats.is_empty()
            || self.swapchain_support.present_modes.is_empty() {
            return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
        }
        if DEVICE_EXTENSIONS.iter().any(|e| !self.extensions.contains(e)) {
            return Err(anyhow!(SuitabilityError("Missing required device extensions.")));
        }

        Ok(())
    }
}
