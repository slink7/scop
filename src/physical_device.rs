

pub const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name
];

use std::collections::HashSet;

use log::*;

use anyhow::{anyhow, Result};

use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::KhrSurfaceExtensionInstanceCommands;

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
    ) -> Result<Self> {
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
        physical_device: vk::PhysicalDevice
    ) -> Result<Self> {
        let properties = instance
            .get_physical_device_queue_family_properties(
                physical_device
            );

        let graphics = properties
            .iter()
            .position(|p| 
                p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        let mut present = None;
        for (index, properties) in properties.iter().enumerate() {
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

pub unsafe fn check_physical_device_extensions(instance: &Instance, physical_device: vk::PhysicalDevice) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError("Missing required device extensions.")))
    }
}

pub unsafe fn check_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice
) -> Result<()> {
    let properties = instance
        .get_physical_device_properties(physical_device);
    if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(anyhow!(SuitabilityError("Only discrete GPUs are supported.")));
    }

    let features = instance
        .get_physical_device_features(physical_device);
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
    }
    if features.geometry_shader != vk::TRUE {
        return Err(anyhow!(SuitabilityError("Missing geometry shader support.")));
    }

    QueueFamilyIndices::get(instance, surface, physical_device)?;

    check_physical_device_extensions(instance, physical_device)?;

    let support = SwapchainSupport::get(instance, surface, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }

    Ok(())
}

pub unsafe fn pick_physical_device(
    instance: &Instance,
    surface: vk::SurfaceKHR
) -> Result<vk::PhysicalDevice> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, surface, physical_device) {
            warn!("Skipping physical device (`{}`): {}", properties.device_name, error);
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            return Ok(physical_device);
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}
#[derive(Clone, Debug, Default)]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<vk::QueueFamilyProperties>,
    pub queue_families_indices: QueueFamilyIndices,
    pub swapchain_support: SwapchainSupport
}

impl PhysicalDevice {
    pub unsafe fn new(
        instance: &vulkanalia::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<Self> {
        let handle = pick_physical_device(instance, surface)?;

        let properties = instance
            .get_physical_device_properties(handle);
        let features = instance
            .get_physical_device_features(handle);
        let memory_properties = instance
            .get_physical_device_memory_properties(handle);
        let queue_families = instance
            .get_physical_device_queue_family_properties(handle);
        let queue_families_indices = QueueFamilyIndices::get(instance, surface, handle)?;
        let swapchain_support = SwapchainSupport::get(instance, surface, handle)?;

        Ok(Self {
            handle,
            properties,
            features,
            memory_properties,
            queue_families,
            queue_families_indices,
            swapchain_support
        })
    }
}
