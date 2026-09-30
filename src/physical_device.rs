

// mod queue_family_indices;

use vulkanalia::prelude::v1_0::*;

#[derive(Clone, Debug, Default)]
pub struct PhysicalDevice {
    pub handle: vk::PhysicalDevice,
    pub properties: vk::PhysicalDeviceProperties,
    pub features: vk::PhysicalDeviceFeatures,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue_families: Vec<vk::QueueFamilyProperties>
}

impl PhysicalDevice {
    pub unsafe fn new(
        instance: &vulkanalia::Instance,
        handle: vk::PhysicalDevice
    ) -> Self {
        let properties = instance.get_physical_device_properties(handle);

        let features = instance.get_physical_device_features(handle);

        let memory_properties = instance.get_physical_device_memory_properties(handle);
        let queue_families = instance.get_physical_device_queue_family_properties(handle);

        Self {
            handle,
            properties,
            features,
            memory_properties,
            queue_families
        }
    }
}
