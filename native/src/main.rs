use cgmath;
use glium::{glutin, implement_vertex, index, uniform, Surface};
use glutin::event::{ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent};
use log::info;
use solver;

const DAM_PARTICLES_X: usize = 10;
const DAM_PARTICLES_Y: usize = 1000;
const MAX_BLOCKS: usize = 50;
const BLOCK_PARTICLES: usize = 500;
const MAX_PARTICLES: usize = DAM_PARTICLES_X * DAM_PARTICLES_Y + MAX_BLOCKS * BLOCK_PARTICLES; // TODO change
const POINT_SIZE: f32 = 7.0;

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
implement_vertex!(Vertex, position);

fn main() -> Result<(), String> {
    env_logger::init();

    let mut sim = solver::State::new();
    sim.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);

    let event_loop = glutin::event_loop::EventLoop::new();
    let size: glutin::dpi::LogicalSize<u32> = (solver::WINDOW_WIDTH, solver::WINDOW_HEIGHT).into();
    let wb = glutin::window::WindowBuilder::new()
        .with_inner_size(size)
        .with_resizable(false)
        .with_title("Position Based Fluid");
    let cb = glutin::ContextBuilder::new();
    let display = glium::Display::new(wb, cb, &event_loop)
        .map_err(|e| format!("Failed to create glium display: {}", e))?;

    let vertex_shader_src = r#"
        #version 140
        uniform mat4 matrix;
        in vec2 position;
        void main() {
            gl_Position = matrix * vec4(position, 0.0, 1.0);
        }
    "#;
    let fragment_shader_src = r#"
        #version 140
        out vec4 f_color;
        void main() {
            f_color = vec4(0.2, 0.6, 1.0, 1.0);
        }
    "#;
    let program =
        glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None)
            .map_err(|e| format!("Failed to parse vertex shader source: {}", e))?;
    let ortho_matrix: [[f32; 4]; 4] = cgmath::ortho(
        0.0,
        solver::WINDOW_WIDTH as f32,
        solver::WINDOW_HEIGHT as f32,
        0.0,
        -1.0,
        1.0,
    )
    .into();
    let uniforms = uniform! {
        matrix: ortho_matrix
    };
    let indices = index::NoIndices(index::PrimitiveType::Points);

    // preallocate vertex buffer
    let vertex_buffer = glium::VertexBuffer::empty_dynamic(&display, MAX_PARTICLES * 2).unwrap();
    let draw_params = glium::DrawParameters {
        polygon_mode: glium::PolygonMode::Point,
        point_size: Some(POINT_SIZE),
        ..Default::default()
    };

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = glutin::event_loop::ControlFlow::Exit;
                    return;
                }
                WindowEvent::KeyboardInput {
                    input:
                        KeyboardInput {
                            virtual_keycode: Some(virtual_code),
                            state,
                            ..
                        },
                    ..
                } => match (virtual_code, state) {
                    (VirtualKeyCode::R, ElementState::Pressed) => {
                        sim.clear();
                        info!("Cleared simulation");
                        sim.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);
                    }
                    (VirtualKeyCode::Space, ElementState::Pressed) => {
                        sim.init_block(BLOCK_PARTICLES);
                    }
                    (VirtualKeyCode::Escape, ElementState::Pressed) => {
                        *control_flow = glutin::event_loop::ControlFlow::Exit;
                        return;
                    }
                    _ => (),
                },
                _ => return,
            },
            Event::NewEvents(cause) => match cause {
                StartCause::Init => (),
                StartCause::Poll => (),
                _ => return,
            },
            _ => return,
        }

        sim.update();

        // draw
        let data: Vec<Vertex> = sim
            .get_positions()
            .iter()
            .map(|p| {
                let mut pp = *p;
                pp.x = solver::DRAW_ORIG.x + pp.x * solver::DRAW_SCALE;
                pp.y = solver::DRAW_ORIG.y - pp.y * solver::DRAW_SCALE;
                Vertex {
                    position: pp.to_array(),
                }
            })
            .collect();
        vertex_buffer.slice(0..data.len()).unwrap().write(&data);

        let mut target = display.draw();
        target.clear_color(0.9, 0.9, 0.9, 1.0);
        target
            .draw(&vertex_buffer, &indices, &program, &uniforms, &draw_params)
            .unwrap();
        target.finish().unwrap();
    });
}
