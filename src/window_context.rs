

use winit::window::Window;

use vulkanalia::{prelude::v1_0::*};

use anyhow::Result;

use crate::vulkan_context::VulkanContext;

pub struct Swapchain {
    
}

#[derive(Debug)]
pub struct WindowContext {
    pub window: Window,
    pub surface: vk::SurfaceKHR
}

impl WindowContext {

    pub unsafe fn new(
        vulkan: &VulkanContext,
        window: Window
    ) -> Result<Self> {
        let surface = vulkanalia::window::create_surface(&vulkan.instance, &window, &window)?;

        Self::from_surface(vulkan, window, surface)
    }

    pub unsafe fn from_surface(
        vulkan: &VulkanContext,
        window: Window,
        surface: vk::SurfaceKHR
    ) -> Result<Self> {
        
        // todo!();

        Ok(Self {window, surface})
    }
}
