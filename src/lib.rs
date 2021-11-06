#![warn(
    unreachable_pub,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    rust_2018_idioms
)]

use solver;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{WebGlBuffer, WebGlProgram, WebGlRenderingContext, WebGlShader};

const DAM_PARTICLES_X: usize = 10;
const DAM_PARTICLES_Y: usize = 1000;
const MAX_BLOCKS: usize = 50;
const BLOCK_PARTICLES: usize = 500;
const MAX_PARTICLES: usize = DAM_PARTICLES_X * DAM_PARTICLES_Y + MAX_BLOCKS * BLOCK_PARTICLES; // TODO change
const POINT_SIZE: f32 = 3.0;

#[wasm_bindgen]
pub struct Simulation {
    context: WebGlRenderingContext,
    position_buffer: WebGlBuffer,
    boundary_buffer: WebGlBuffer,
    state: solver::State,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas: &web_sys::HtmlCanvasElement) -> Result<Simulation, JsValue> {
        let (context, position_buffer, boundary_buffer) = init_webgl(canvas)?;
        let mut state = solver::State::new();
        state.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);
        generate_boundary_vertex_array(&context, &boundary_buffer, state.get_boundaries());
        Ok(Simulation {
            context,
            position_buffer,
            boundary_buffer,
            state,
        })
    }

    #[wasm_bindgen]
    pub fn get_num_particles(&self) -> usize {
        self.state.num_particles
    }

    #[wasm_bindgen]
    pub fn set_viscosity(&mut self, viscosity: f32) {
        self.state.set_viscosity(viscosity);
    }

    #[wasm_bindgen]
    pub fn set_solver_substeps(&mut self, num_substeps: usize) {
        self.state.set_solver_substeps(num_substeps);
    }

    #[wasm_bindgen]
    pub fn step(&mut self) {
        self.state.update();
        self.draw();
    }

    #[wasm_bindgen]
    pub fn add_block(&mut self) {
        if self.get_num_particles() < MAX_PARTICLES - BLOCK_PARTICLES {
            self.state.init_block(BLOCK_PARTICLES);
        }
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.state.clear();
        self.state.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);
    }

    fn draw(&self) {
        self.context.clear(WebGlRenderingContext::COLOR_BUFFER_BIT);

        /*
        // lines for boundaries
        self.context.bind_buffer(
            WebGlRenderingContext::ARRAY_BUFFER,
            Some(&self.boundary_buffer),
        );
        let vert_count = 12;
        self.context
            .draw_arrays(WebGlRenderingContext::TRIANGLES, 0, vert_count);
        */

        // particles
        self.context.bind_buffer(
            WebGlRenderingContext::ARRAY_BUFFER,
            Some(&self.position_buffer),
        );
        let vertices: Vec<f32> = self
            .state
            .get_positions()
            .iter()
            .map(|p| {
                let mut pp = *p;
                pp.x = solver::DRAW_ORIG.x + pp.x * solver::DRAW_SCALE;
                pp.y = solver::DRAW_ORIG.y - pp.y * solver::DRAW_SCALE;
                pp.to_array()
            })
            .flatten()
            .collect();
        unsafe {
            // Note that `Float32Array::view` is somewhat dangerous (hence the
            // `unsafe`!). This is creating a raw view into our module's
            // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
            // (aka do a memory allocation in Rust) it'll cause the buffer to change,
            // causing the `Float32Array` to be invalid.
            //
            // As a result, after `Float32Array::view` we have to be very careful not to
            // do any memory allocations before it's dropped.
            let positions_array_buf_view = js_sys::Float32Array::view(&vertices);

            self.context.buffer_sub_data_with_i32_and_array_buffer_view(
                WebGlRenderingContext::ARRAY_BUFFER,
                0,
                &positions_array_buf_view,
            );
        }

        let vert_count = self.state.num_particles as i32;
        self.context
            .draw_arrays(WebGlRenderingContext::POINTS, 0, vert_count);
    }
}

fn generate_boundary_vertex_array(
    context: &WebGlRenderingContext,
    buffer: &WebGlBuffer,
    boundaries: Vec<[f32; 4]>,
) {
    let boundaries: Vec<f32> = boundaries
        .iter()
        .map(|p| {
            // specified as [x0, x0+width, y0, y0+height]
            let x0 = solver::DRAW_ORIG.x + p[0] * solver::DRAW_SCALE; // x0
            let y0 = solver::DRAW_ORIG.y - p[2] * solver::DRAW_SCALE;
            let width = (p[1] - p[0]) * solver::DRAW_SCALE;
            let height = (p[3] - p[2]) * solver::DRAW_SCALE;
            [
                x0,
                y0,
                x0 + width,
                y0,
                x0 + width,
                y0 - height,
                x0,
                x0 - height,
            ]
        })
        .flatten()
        .collect();

    unsafe {
        let boundaries_array_buf_view = js_sys::Float32Array::view(&boundaries);

        context.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, Some(&buffer));
        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &boundaries_array_buf_view,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }
}

fn init_webgl(
    canvas: &web_sys::HtmlCanvasElement,
) -> Result<(WebGlRenderingContext, WebGlBuffer, WebGlBuffer), JsValue> {
    // set up canvas and webgl context handle
    canvas.set_width(solver::WINDOW_WIDTH);
    canvas.set_height(solver::WINDOW_HEIGHT);
    let context = canvas
        .get_context("webgl")?
        .unwrap()
        .dyn_into::<WebGlRenderingContext>()?;

    context.viewport(
        0,
        0,
        solver::WINDOW_WIDTH as i32,
        solver::WINDOW_HEIGHT as i32,
    );
    context.clear_color(0.9, 0.9, 0.9, 1.0);

    let vert_shader = compile_shader(
        &context,
        WebGlRenderingContext::VERTEX_SHADER,
        format!(
            r##"
        uniform mat4 matrix;
        attribute vec2 position;
        void main() {{
            gl_PointSize = {:.1};
            gl_Position = matrix * vec4(position, 0.0, 1.0);
        }}
        "##,
            POINT_SIZE
        )
        .as_str(),
    )?;

    let frag_shader = compile_shader(
        &context,
        WebGlRenderingContext::FRAGMENT_SHADER,
        r##"
        precision mediump float;
        void main() {{
            gl_FragColor = vec4(0.2, 0.6, 1.0, 1.0);
        }}
        "##,
    )?;
    let program = link_program(&context, &vert_shader, &frag_shader)?;
    context.use_program(Some(&program));

    // uniforms
    let uniform_location = context
        .get_uniform_location(&program, "matrix")
        .expect("Unable to get shader projection matrix uniform location");
    let ortho_matrix = cgmath::ortho(
        0.0,
        solver::WINDOW_WIDTH as f32,
        solver::WINDOW_HEIGHT as f32,
        0.0,
        -1.0,
        1.0,
    );
    let ortho_matrix_flattened_ref: &[f32; 16] = ortho_matrix.as_ref();
    context.uniform_matrix4fv_with_f32_array(
        Some(&uniform_location),
        false,
        ortho_matrix_flattened_ref,
    );

    // attributes
    let position_attribute_location = context.get_attrib_location(&program, "position");
    let boundary_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    context.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, Some(&boundary_buffer));
    context.vertex_attrib_pointer_with_i32(0, 2, WebGlRenderingContext::FLOAT, false, 0, 0);
    context.enable_vertex_attrib_array(position_attribute_location as u32);

    let position_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    context.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, Some(&position_buffer));
    context.vertex_attrib_pointer_with_i32(0, 2, WebGlRenderingContext::FLOAT, false, 0, 0);
    context.enable_vertex_attrib_array(position_attribute_location as u32);

    // allocate vertex buffer initial state
    let zeroed = vec![0.0; MAX_PARTICLES * 2];
    unsafe {
        let positions_array_buf_view = js_sys::Float32Array::view(&zeroed);

        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGlRenderingContext::DYNAMIC_DRAW,
        );
    }
    Ok((context, position_buffer, boundary_buffer))
}

fn compile_shader(
    context: &WebGlRenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGlRenderingContext::COMPILE_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(shader)
    } else {
        Err(context
            .get_shader_info_log(&shader)
            .unwrap_or_else(|| String::from("Unknown error creating shader")))
    }
}

fn link_program(
    context: &WebGlRenderingContext,
    vert_shader: &WebGlShader,
    frag_shader: &WebGlShader,
) -> Result<WebGlProgram, String> {
    let program = context
        .create_program()
        .ok_or_else(|| String::from("Unable to create shader object"))?;

    context.attach_shader(&program, vert_shader);
    context.attach_shader(&program, frag_shader);
    context.link_program(&program);

    if context
        .get_program_parameter(&program, WebGlRenderingContext::LINK_STATUS)
        .as_bool()
        .unwrap_or(false)
    {
        Ok(program)
    } else {
        Err(context
            .get_program_info_log(&program)
            .unwrap_or_else(|| String::from("Unknown error creating program object")))
    }
}
