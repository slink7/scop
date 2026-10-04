
use std::fs::File;

use anyhow::Result;

use png::Reader;

use vulkanalia::prelude::v1_0::*;

use crate::image::Image;

fn load_png_data(
    path: &str
) -> Result<(Vec<u8>, u32, u32)> {
    let image = File::open(path)?;

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;

    let mut pixels = vec![0; reader.info().raw_bytes()];

    reader.next_frame(&mut pixels)?;

    let (width, height) = reader.info().size();

    Ok((pixels, width, height))
}

#[derive(Default)]
pub struct Texture {
    pub image: Image,
    pub sampler: vk::Sampler
}
