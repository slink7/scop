#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;

use anyhow::{Result};

use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent, ElementState};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::event_loop::EventLoop;
use winit::window::{WindowBuilder};

mod model;
mod vertex;
mod app;
mod physical_device;
mod vulkan_context;

use crate::app::App;

fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Winit Window yay")
        .with_inner_size(LogicalSize::new(768, 768))
        .build(&event_loop)?;

    print!("Pre create()");
    let mut app = unsafe { App::create(&window)? };
    let mut minimized = false;
    event_loop.run(move |event, elwt| {
        match event {
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed {
                        match event.physical_key {
                            PhysicalKey::Code(KeyCode::ArrowLeft) if app.models > 1 => app.models -= 1,
                            PhysicalKey::Code(KeyCode::ArrowRight) if app.models < 4 => app.models += 1,
                            _ => {}
                        }
                    }
                },
                WindowEvent::Resized(size) => {
                    if size.width == 0 || size.height == 0 {
                        minimized = true;
                    } else {
                        minimized = false;
                        app.resized = true
                    }
                },
                WindowEvent::RedrawRequested if !elwt.exiting() && !minimized => unsafe {
                    if let Err(e) = app.render(&window) {
                        eprintln!("render error: {e:?}");
                        elwt.exit();
                    }
                },
                WindowEvent::CloseRequested => {
                    elwt.exit();
                    unsafe {
                        app.destroy();
                    }
                }
                _ => {}
            }
            _ => {}
        }
    })?;

    Ok(())
}
