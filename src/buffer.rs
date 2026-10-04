use anyhow::Result;

use vulkanalia::vk::{self, DeviceV1_0, HasBuilder};

use crate::vulkan_context::VulkanContext;

#[derive(Default, Debug)]
pub struct Buffer {
    pub handle: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize
}

impl Buffer {
    pub unsafe fn new(
        vulkan: &VulkanContext,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        properties: vk::MemoryPropertyFlags
    ) -> Result<Self> {
        
        let buffer_info = vk::BufferCreateInfo::builder()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let handle = vulkan.device.device.create_buffer(&buffer_info, None)?;

        let requirements = vulkan.device.device.get_buffer_memory_requirements(handle);

        let memory_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(requirements.size)
            .memory_type_index(
                vulkan.physical_device.memory_type_index(
                    properties,
                    requirements
                )?
            );

        let memory = vulkan.device.device.allocate_memory(&memory_info, None)?;
    
        vulkan.device.device.bind_buffer_memory(handle, memory, 0)?;

        Ok(Self {
            handle,
            memory,
            size
        })
    }

    pub unsafe fn destroy(
        &mut self,
        vulkan: &VulkanContext
    ) {
        vulkan.device.device.destroy_buffer(self.handle, None);
        vulkan.device.device.free_memory(self.memory, None);
        *self = Self::default();
    }
}
