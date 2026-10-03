use anyhow::{Result, anyhow};
use std::os::raw::c_void;
use std::ffi::CStr;
use std::collections::HashSet;

use log::*;

use vulkanalia::{loader::{LIBRARY, LibloadingLoader}, prelude::v1_0::*, vk::KhrSurfaceExtensionInstanceCommands};
use vulkanalia::vk::{ExtDebugUtilsExtensionInstanceCommands};

use winit::window::Window;

use crate::{app::VALIDATION_ENABLED, window_context::WindowContext};
use crate::physical_device::{
        PhysicalDevice
};

// const VALIDATION_ENABLED: bool = false;//cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName = 
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");
const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[
    vk::KHR_SWAPCHAIN_EXTENSION.name
];

pub extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        debug!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

pub unsafe fn create_debug_messenger(
    instance: &Instance
) -> Result<vk::DebugUtilsMessengerEXT> {
    if !VALIDATION_ENABLED {
        return Err(anyhow!("Validation not enabled"));
    }
    let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION |
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        )
        .user_callback(Some(debug_callback));

    Ok(instance.create_debug_utils_messenger_ext(&debug_info, None)?)
}

pub unsafe fn create_instance(window: &Window, entry: &Entry)
    -> Result<Instance>
{
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"scop\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"bababooey engine\0")
        .engine_version(vk::make_version(0, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    let mut extensions = vulkanalia::window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    let flags = vk::InstanceCreateFlags::empty();

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL |
            vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION |
            vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        )
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        info = info.push_next(&mut debug_info);
    }

    let instance = entry.create_instance(&info, None)?;

    Ok(instance)
}

pub unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    physical_device: &PhysicalDevice
) -> Result<Device> {
    let indices = physical_device.queue_families_indices;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);

    let queue_priorities = &[1.0];
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    let extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true)
        .sample_rate_shading(false);
    
    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(physical_device.handle, &info, None)?;
    Ok(device)
}

#[derive(Clone, Debug)]
pub struct DeviceContext {
    pub device: vulkanalia::Device,
    pub graphics_queue: vk::Queue,
    pub present_queue: vk::Queue
}

impl DeviceContext {
    pub unsafe fn new(
        entry: &Entry,
        instance: &Instance,
        physical_device: &PhysicalDevice
    ) -> Result<Self> {
        
        let device = create_logical_device(entry, instance, physical_device)?;

        let indices = physical_device.queue_families_indices;
        let graphics_queue = device.get_device_queue(indices.graphics, 0);
        let present_queue = device.get_device_queue(indices.present, 0);
        Ok(Self {
            device,
            graphics_queue,
            present_queue
        })
    }

    pub unsafe fn destroy(&mut self) {
        self.device.destroy_device(None);
    }
}

#[derive(Clone, Debug)]
pub struct VulkanContext {
    pub instance: vulkanalia::Instance,
    pub messenger: vk::DebugUtilsMessengerEXT,
    pub physical_device: PhysicalDevice,
    pub device: DeviceContext,
    pub surface: vk::SurfaceKHR,
    pub entry: Entry
}

impl VulkanContext {
    pub unsafe fn new(window: Window) -> Result<(Self, WindowContext)> {
        let loader = LibloadingLoader::new(LIBRARY)?;
        let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
        let instance = create_instance(&window, &entry)?;
        let messenger = create_debug_messenger(&instance).unwrap_or_default();
        let surface = vulkanalia::window::create_surface(&instance, &window, &window)?;
        let physical_device = PhysicalDevice::new(&instance, surface)?;
        let device = DeviceContext::new(&entry, &instance, &physical_device)?;


        let vulkan = Self {
            instance,
            messenger,
            physical_device,
            device,
            surface,
            entry
        };
        let window = WindowContext::from_surface(&vulkan, window, surface)?;

        Ok((vulkan, window))
    }

    pub unsafe fn destroy(&mut self) {

        self.device.destroy();

        if VALIDATION_ENABLED {
            self.instance.destroy_debug_utils_messenger_ext(self.messenger, None);
        }

        self.instance.destroy_surface_khr(self.surface, None);

        self.instance.destroy_instance(None);
    }
}
