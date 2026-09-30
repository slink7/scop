
use anyhow::Result;
use cgmath::{vec2, vec3};
use std::{
    collections::HashMap,
    fs::File,
    io::BufReader
};

use crate::vertex::Vertex;

#[derive(Clone, Debug, Default)]
pub struct Model {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl Model {
    pub fn load(path: &str) -> Result<Self> {
        let mut reader = BufReader::new(File::open(path)?);

        let (models, _) = tobj::load_obj_buf(
            &mut reader,
            &tobj::LoadOptions {
                triangulate: true,
                ..Default::default()
            },
            |_| Ok(Default::default())
        )?;

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut unique_vertices = HashMap::new();

        for model in &models {
            for index in &model.mesh.indices {
                let pos_offset = (3 * index) as usize;
                let tex_coord_offset = (2 * index) as usize;

                let vertex = Vertex {
                    pos: vec3(
                        model.mesh.positions[pos_offset + 0],
                        model.mesh.positions[pos_offset + 1],
                        model.mesh.positions[pos_offset + 2]
                    ),
                    color: vec3(1.0, 1.0, 1.0),
                    tex_coord: vec2(
                        model.mesh.texcoords[tex_coord_offset],
                        1.0 - model.mesh.texcoords[tex_coord_offset + 1]
                    )
                };

                if let Some(index) = unique_vertices.get(&vertex) {
                    indices.push(*index as u32);
                } else {
                    let index = vertices.len();

                    unique_vertices.insert(vertex, index);
                    vertices.push(vertex);
                    indices.push(index as u32);
                }
            }
        }
        Ok(Self {
            vertices,
            indices
        })
    }
}
