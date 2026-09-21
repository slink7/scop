#![allow(
    dead_code,
    unsafe_op_in_unsafe_fn,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use anyhow::{anyhow, Result};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};
use log::*;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::window as vk_window;
use vulkanalia::prelude::v1_0::*;

unsafe fn creat_instance(window: &Window, entry: &Entry) -> Result<Instance> {
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"scop\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"bababooey engine\0")
        .engine_version(vk::make_version(0, 0, 0))
        .api_version(vk::make_version(1, 0, 0,));

    let extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    let info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_extension_names(&extensions);

    Ok(entry.create_instance(&info, None)?);
}

fn main() -> Result<()> {
    pretty_env_logger::init();

    let event_loop = EventLoop::new()?;
    let window = WindowBuilder::new()
        .with_title("Winit Window yay")
        .with_inner_size(LogicalSize::new(1024, 768))
        .build(&event_loop)?;

    let mut app = unsafe { App::create(&window)? };
    event_loop.run(move |event, elwt| {
        match event {
            Event::AboutToWait => window.request_redraw(),
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::RedrawRequested if !elwt.exiting() => unsafe {
                    app.render(&window)
                }.unwrap(),
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

#[derive(Clone, Debug)]
struct App {
    it: i32
}

impl App {
    
    unsafe fn create(window: &Window) -> Result<Self> {
        Ok(Self { it: 5 })
    }

    unsafe fn render(&mut self, window: &Window) -> Result<()> {
        print!("{}\n", self.it);
        self.it += 1;
        Ok(())
    }

    unsafe fn destroy(&mut self) {
    }

}

#[derive(Clone, Debug, Default)]
struct AppData {}
