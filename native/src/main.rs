use std::borrow::Cow;

use cgmath;
use glam::Vec2;
use glium::{glutin, implement_vertex, index, uniform, Surface, VertexFormat};
use glutin::event::{ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent};
use log::info;

use solver;

const DAM_PARTICLES_X: usize = 10;
const DAM_PARTICLES_Y: usize = 1000;
const BLOCK_PARTICLES: usize = 500;
const MAX_PARTICLES: usize = solver::MAX_PARTICLES;
const POINT_SIZE: f32 = 7.5;
const BOUNDARY_COLOR: [f32; 4] = [112.0 / 255.0, 128.0 / 255.0, 144.0 / 255.0, 1.0]; // #708090
const PARTICLE_COLOR: [f32; 4] = [65.0 / 255.0, 105.0 / 255.0, 1.0, 1.0]; // #4169E1

#[derive(Copy, Clone)]
struct Vertex {
    position: [f32; 2],
}
implement_vertex!(Vertex, position);

fn main() -> Result<(), String> {
    env_logger::init();

    let mut sim = solver::State::new();
    sim.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);
    info!("Initialized dam break with {} particles", sim.num_particles);

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
        uniform mat4 projection_matrix;
        uniform mat4 view_matrix;
        in vec2 position;
        void main() {
            gl_Position = projection_matrix *  view_matrix * vec4(position, 0.0, 1.0);
        }
    "#;
    let fragment_shader_src = r#"
        #version 140
        uniform vec4 draw_color;
        out vec4 f_color;
        void main() {
            f_color = draw_color;
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
    let view_matrix: [[f32; 4]; 4] = [
        [solver::DRAW_SCALE, 0.0, 0.0, 0.0],
        [0.0, -solver::DRAW_SCALE, 0.0, 0.0], // flip y coordinate from solver
        [0.0, 0.0, solver::DRAW_SCALE, 0.0],
        [solver::DRAW_ORIG.x, solver::DRAW_ORIG.y, 0.0, 1.0],
    ];

    // prepopulate boundary geometry and draw configuration
    let boundaries: Vec<Vertex> = sim
        .get_boundaries()
        .iter()
        .map(|p| {
            // specified as [x0, x0+width, y0, y0+height]
            let x = p[0];
            let y = p[2];
            let w = p[1] - p[0];
            let h = p[3] - p[2];
            // form a rectangle using two triangles, three vertices each
            [
                Vertex { position: [x, y] },
                Vertex {
                    position: [x + w, y],
                },
                Vertex {
                    position: [x + w, y + h],
                },
                Vertex { position: [x, y] },
                Vertex {
                    position: [x, y + h],
                },
                Vertex {
                    position: [x + w, y + h],
                },
            ]
        })
        .flatten()
        .collect();
    let boundary_vertex_buffer =
        glium::VertexBuffer::empty_immutable(&display, boundaries.len())
            .map_err(|e| format!("Failed to create boundary vertex buffer: {}", e))?;
    let boundary_uniforms = uniform! {
        projection_matrix: ortho_matrix,
        view_matrix: view_matrix,
        draw_color: BOUNDARY_COLOR,
    };
    let boundary_indices = glium::index::NoIndices(glium::index::PrimitiveType::TrianglesList);
    let boundary_draw_params = glium::DrawParameters {
        polygon_mode: glium::PolygonMode::Fill,
        ..Default::default()
    };
    boundary_vertex_buffer.write(&boundaries);

    // preallocate particle vertex buffer
    let empty_buffer = vec![Vec2::ZERO; MAX_PARTICLES];
    let bindings: VertexFormat = Cow::Owned(vec![(
        Cow::Borrowed("position"),
        2 * std::mem::size_of::<f32>(),
        0,
        glium::vertex::AttributeType::F32F32,
        false,
    )]);
    let particle_vertex_buffer = unsafe {
        glium::VertexBuffer::new_raw_dynamic(
            &display,
            &empty_buffer,
            bindings,
            2 * std::mem::size_of::<f32>(),
        )
        .map_err(|e| format!("Failed to create particle vertex buffer: {}", e))?
    };
    let particle_uniforms = uniform! {
        projection_matrix: ortho_matrix,
        view_matrix: view_matrix,
        draw_color: PARTICLE_COLOR,
    };
    let partcile_indices = index::NoIndices(index::PrimitiveType::Points);
    let particle_draw_params = glium::DrawParameters {
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
                        particle_vertex_buffer.invalidate();
                        particle_vertex_buffer.write(&vec![Vec2::ZERO; MAX_PARTICLES]);
                        sim.clear();
                        info!("Cleared simulation");
                        sim.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);
                        info!("Initialized dam break with {} particles", sim.num_particles);
                    }
                    (VirtualKeyCode::Space, ElementState::Pressed) => {
                        if sim.init_block(BLOCK_PARTICLES) {
                            info!(
                                "Initialized block of {} particles, new total {}",
                                BLOCK_PARTICLES, sim.num_particles
                            );
                        } else {
                            info!("Max particles reached");
                        }
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

        particle_vertex_buffer
            .slice(0..sim.num_particles)
            .unwrap()
            .write(sim.get_positions()); // unwrap is safe due to preallocated known length

        let mut target = display.draw();
        target.clear_color(0.9, 0.9, 0.9, 1.0);

        // draw boundaries
        target
            .draw(
                &boundary_vertex_buffer,
                &boundary_indices,
                &program,
                &boundary_uniforms,
                &boundary_draw_params,
            )
            .unwrap();

        // draw particles
        target
            .draw(
                &particle_vertex_buffer,
                &partcile_indices,
                &program,
                &particle_uniforms,
                &particle_draw_params,
            )
            .unwrap();
        target.finish().unwrap();
    });
}
