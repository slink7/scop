
use anyhow::Result;

use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{Extent2D};

use crate::physical_device::{self, PhysicalDevice};
use crate::vulkan_context::VulkanContext;

pub unsafe fn create_image(
    device: &Device,
    physical_device: &PhysicalDevice,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(mip_levels)
        .array_layers(1)
        .format(format)
        .tiling(tiling)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .usage(usage)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .samples(samples)
        .flags(vk::ImageCreateFlags::empty());

    let image = device.create_image(&info, None)?;

    let requirements = device.get_image_memory_requirements(image);

    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(
            physical_device.memory_type_index(
                properties,
                requirements
            )?
        );

    let image_memory = device.allocate_memory(&info, None)?;

    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

pub unsafe fn create_image_view(device: &Device, image: vk::Image, format: vk::Format, aspects: vk::ImageAspectFlags, mip_levels: u32) -> Result<vk::ImageView> {

    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .subresource_range(subresource_range);

    Ok(device.create_image_view(&info, None)?)
}

#[derive(Debug, Default)]
pub struct Image {
    pub image: vk::Image,
    pub memory: vk::DeviceMemory,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub mip_level: u32
}

impl Image {
    pub unsafe fn new(
        vulkan: &VulkanContext,
        size: Extent2D,
        mip_level: u32,
        samples: vk::SampleCountFlags,
        format: vk::Format,
        usage: vk::ImageUsageFlags,
        aspects: vk::ImageAspectFlags
    ) -> Result<Self> {

        let (image, memory) = create_image(
            &vulkan.device.device,
            &vulkan.physical_device,
            size.width,
            size.height,
            mip_level,
            samples,
            format,
            vk::ImageTiling::OPTIMAL,
            usage,
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        )?;

        let view = create_image_view(
            &vulkan.device.device,
            image,
            format,
            aspects,
            1,
        )?;

        Ok(Self {
            image,
            memory,
            view,
            format,
            mip_level
        })
    }

    pub unsafe fn new_color(
        vulkan: &VulkanContext,
        size: Extent2D,
        format: vk::Format
    ) -> Result<Self> {
        Self::new(
            vulkan,
            size,
            1,
            vulkan.physical_device.max_msaa_samples,
            format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT |
            vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
            vk::ImageAspectFlags::COLOR
        )
    }

    pub unsafe fn new_depth(
        vulkan: &VulkanContext,
        size: Extent2D,
    ) -> Result<Self> {
        Self::new(
            vulkan,
            size,
            1,
            vulkan.physical_device.max_msaa_samples,
            vulkan.physical_device.depth_format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
            vk::ImageAspectFlags::DEPTH
        )
    }

    pub unsafe fn destroy(
        &mut self,
        vulkan: &VulkanContext
    ) {
        vulkan.device.device.destroy_image_view(self.view, None);
        vulkan.device.device.free_memory(self.memory, None);
        vulkan.device.device.destroy_image(self.image, None);

        *self = Self::default();
    }
}
