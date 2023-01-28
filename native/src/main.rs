#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use std::borrow::Cow;

use glam::Vec2;
use glium::{glutin, implement_vertex, index, uniform, Surface, VertexFormat};
use glutin::event::{ElementState, Event, KeyboardInput, StartCause, VirtualKeyCode, WindowEvent};
use log::info;

// default scene
const DAM_PARTICLES_X: usize = 10;
const DAM_PARTICLES_Y: usize = 1000;

const BLOCK_PARTICLES: usize = 500;
const MAX_PARTICLES: usize = solver::MAX_PARTICLES;
const POINT_SIZE: f32 = 7.5;

#[derive(Copy, Clone)]
struct Vertex {
    in_position: [f32; 2],
}
implement_vertex!(Vertex, in_position);

#[allow(clippy::too_many_lines)]
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
        .map_err(|e| format!("Failed to create glium display: {e}"))?;

    let vertex_shader_src = r#"
        #version 140
        precision mediump float;
        const vec4 particle_color_1 = vec4(0.2549019608, 0.4117647059, 1.0, 1.0); // #4169E1
        const vec4 particle_color_2 = vec4(1.0, 0.2549019608, 0.2980392157, 1.0); // #E1414C

        uniform mat4 u_projection_matrix;
        uniform mat4 u_view_matrix;
        uniform int u_draw_mode_single_color;
        in vec2 in_position;
        out vec4 frag_color;

        void main() {
            gl_Position = u_projection_matrix * u_view_matrix * vec4(in_position, 0.0, 1.0);
            if (u_draw_mode_single_color == 1 || int(floor(float(gl_VertexID) / 1000.0)) % 2 == 0) {
                frag_color = particle_color_1;
            } else {
                frag_color = particle_color_2;
            }
        }
    "#;
    let fragment_shader_src = r#"
        #version 140
        precision mediump float;
        const vec4 boundary_color = vec4(0.4392156863, 0.5019607843, 0.5647058824, 1.0); // #708090

        uniform int u_draw_mode_boundary;
        in vec4 frag_color;
        out vec4 out_color;

        void main() {
            if (u_draw_mode_boundary == 1) {
                out_color = boundary_color;
            } else {
                out_color = frag_color;
            }
        }
    "#;
    let program =
        glium::Program::from_source(&display, vertex_shader_src, fragment_shader_src, None)
            .map_err(|e| format!("Failed to parse vertex shader source: {e}"))?;
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
        .flat_map(|p| {
            // specified as [x0, x0+width, y0, y0+height]
            let x = p[0];
            let y = p[2];
            let w = p[1] - p[0];
            let h = p[3] - p[2];
            // form a rectangle using two triangles, three vertices each
            [
                Vertex {
                    in_position: [x, y],
                },
                Vertex {
                    in_position: [x + w, y],
                },
                Vertex {
                    in_position: [x + w, y + h],
                },
                Vertex {
                    in_position: [x, y],
                },
                Vertex {
                    in_position: [x, y + h],
                },
                Vertex {
                    in_position: [x + w, y + h],
                },
            ]
        })
        .collect();
    let boundary_vertex_buffer =
        glium::VertexBuffer::empty_immutable(&display, boundaries.len())
            .map_err(|e| format!("Failed to create boundary vertex buffer: {e}"))?;
    let boundary_uniforms = uniform! {
        u_projection_matrix: ortho_matrix,
        u_view_matrix: view_matrix,
        u_draw_mode_boundary: 1,
        u_draw_mode_single_color: 0,
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
        Cow::Borrowed("in_position"),
        0,
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
        .map_err(|e| format!("Failed to create particle vertex buffer: {e}"))?
    };
    let particle_uniforms_base = uniform! {
        u_projection_matrix: ortho_matrix,
        u_view_matrix: view_matrix,
        u_draw_mode_boundary: 0,
    };
    let mut draw_single_color = false;
    let mut particle_uniforms =
        particle_uniforms_base.add("u_draw_mode_single_color", draw_single_color.into());
    let partcile_indices = index::NoIndices(index::PrimitiveType::Points);
    let particle_draw_params = glium::DrawParameters {
        polygon_mode: glium::PolygonMode::Point,
        point_size: Some(POINT_SIZE),
        ..Default::default()
    };

    event_loop.run(move |event, _, control_flow| {
        let mut reset = |x: usize, y: usize| {
            particle_vertex_buffer.invalidate();
            particle_vertex_buffer.write(&vec![Vec2::ZERO; MAX_PARTICLES]);
            sim.clear();
            info!("Cleared simulation");
            sim.init_dam_break(x, y);
            info!("Initialized dam break with {} particles", sim.num_particles);
        };

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
                    (VirtualKeyCode::Key1, ElementState::Pressed) => {
                        reset(10, 1000);
                    }
                    (VirtualKeyCode::Key2, ElementState::Pressed) => {
                        reset(10, 200);
                    }
                    (VirtualKeyCode::Key3, ElementState::Pressed) => {
                        reset(40, 100);
                    }
                    (VirtualKeyCode::Key4, ElementState::Pressed) => {
                        reset(100, 100);
                    }
                    (VirtualKeyCode::S, ElementState::Pressed) => {
                        draw_single_color = !draw_single_color;
                        particle_uniforms = particle_uniforms_base.add(
                            "u_draw_mode_single_color",
                            Into::<i32>::into(draw_single_color),
                        );
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
                StartCause::Init | StartCause::Poll => (),
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
                boundary_indices,
                &program,
                &boundary_uniforms,
                &boundary_draw_params,
            )
            .unwrap();

        // draw particles
        target
            .draw(
                &particle_vertex_buffer,
                partcile_indices,
                &program,
                &particle_uniforms,
                &particle_draw_params,
            )
            .unwrap();
        target.finish().unwrap();
    });
}
