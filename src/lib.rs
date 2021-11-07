#![warn(
    unreachable_pub,
    trivial_casts,
    trivial_numeric_casts,
    unused_extern_crates,
    rust_2018_idioms
)]

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    WebGlBuffer, WebGlProgram, WebGlRenderingContext, WebGlShader, WebGlUniformLocation,
};

const DAM_PARTICLES_X: usize = 10;
const DAM_PARTICLES_Y: usize = 1000;
const BLOCK_PARTICLES: usize = 400;
const MAX_PARTICLES: usize = solver::MAX_PARTICLES;
const POINT_SIZE: f32 = 3.0;
const BOUNDARY_COLOR: [f32; 4] = [112.0 / 255.0, 128.0 / 255.0, 144.0 / 255.0, 1.0]; // #708090
const PARTICLE_COLOR: [f32; 4] = [65.0 / 255.0, 105.0 / 255.0, 1.0, 1.0]; // #4169E1

#[wasm_bindgen]
pub struct Simulation {
    renderer: RenderState,
    state: solver::State,
}

struct RenderState {
    context: WebGlRenderingContext,
    boundary_buffer: WebGlBuffer,
    particle_buffer: WebGlBuffer,
    draw_color_uniform: WebGlUniformLocation,
    position_attrib_location: u32,
}

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new(canvas: &web_sys::HtmlCanvasElement) -> Result<Simulation, JsValue> {
        let mut state = solver::State::new();
        state.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);
        let renderer = init_webgl(canvas, state.get_boundaries())?;
        Ok(Simulation { renderer, state })
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
        self.state.init_block(BLOCK_PARTICLES);
    }

    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.state.clear();
        self.state.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);
    }

    fn draw(&self) {
        self.renderer
            .context
            .clear(WebGlRenderingContext::COLOR_BUFFER_BIT);

        // draw boundaries
        set_buffers_and_attributes(
            &self.renderer.context,
            &self.renderer.boundary_buffer,
            self.renderer.position_attrib_location,
        );
        self.renderer
            .context
            .uniform4fv_with_f32_array(Some(&self.renderer.draw_color_uniform), &BOUNDARY_COLOR);
        self.renderer
            .context
            .draw_arrays(WebGlRenderingContext::TRIANGLES, 0, 12);

        // draw particles
        set_buffers_and_attributes(
            &self.renderer.context,
            &self.renderer.particle_buffer,
            self.renderer.position_attrib_location,
        );
        let vertices: Vec<f32> = self
            .state
            .get_positions()
            .iter()
            .map(|p| p.to_array())
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

            self.renderer
                .context
                .buffer_sub_data_with_i32_and_array_buffer_view(
                    WebGlRenderingContext::ARRAY_BUFFER,
                    0,
                    &positions_array_buf_view,
                );
        }
        self.renderer
            .context
            .uniform4fv_with_f32_array(Some(&self.renderer.draw_color_uniform), &PARTICLE_COLOR);
        self.renderer.context.draw_arrays(
            WebGlRenderingContext::POINTS,
            0,
            self.state.num_particles as i32,
        );
    }
}

fn init_webgl(
    canvas: &web_sys::HtmlCanvasElement,
    boundaries: Vec<[f32; 4]>,
) -> Result<RenderState, JsValue> {
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
        uniform mat4 projection_matrix;
        uniform mat4 view_matrix;
        attribute vec2 position;
        void main() {{
            gl_PointSize = {:.1};
            gl_Position = projection_matrix * view_matrix * vec4(position, 0.0, 1.0);
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
        uniform vec4 draw_color;
        void main() {{
            gl_FragColor = draw_color;
        }}
        "##,
    )?;
    let program = link_program(&context, &vert_shader, &frag_shader)?;
    context.use_program(Some(&program));

    // set shader matrix uniforms
    let projection_uniform = context
        .get_uniform_location(&program, "projection_matrix")
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
        Some(&projection_uniform),
        false,
        ortho_matrix_flattened_ref,
    );
    let view_uniform = context
        .get_uniform_location(&program, "view_matrix")
        .expect("Unable to get shader view matrix uniform location");
    let view_matrix: [f32; 16] = [
        solver::DRAW_SCALE,
        0.0,
        0.0,
        0.0,
        0.0,
        -solver::DRAW_SCALE, // flip y coordinate from solver
        0.0,
        0.0,
        0.0,
        0.0,
        solver::DRAW_SCALE,
        0.0,
        solver::DRAW_ORIG.x,
        solver::DRAW_ORIG.y,
        0.0,
        1.0,
    ];
    context.uniform_matrix4fv_with_f32_array(Some(&view_uniform), false, &view_matrix);
    let draw_color_uniform = context
        .get_uniform_location(&program, "draw_color")
        .expect("Unable to get fragment draw color uniform location");
    let position_attrib_location = context.get_attrib_location(&program, "position") as u32;

    // prepopulate boundary geometry
    let boundaries: Vec<f32> = boundaries
        .iter()
        .map(|p| {
            // specified as [x0, x0+width, y0, y0+height]
            let x = p[0];
            let y = p[2];
            let w = p[1] - p[0];
            let h = p[3] - p[2];
            // form a rectangle using two triangles, three vertices each
            [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y],
                [x, y + h],
                [x + w, y + h],
            ]
        })
        .flatten()
        .flatten()
        .collect();
    let boundary_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    set_buffers_and_attributes(&context, &boundary_buffer, position_attrib_location);
    unsafe {
        let boundaries_array_buf_view = js_sys::Float32Array::view(&boundaries);

        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &boundaries_array_buf_view,
            WebGlRenderingContext::STATIC_DRAW,
        );
    }

    // preallocate particle vertex buffer
    let particle_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    set_buffers_and_attributes(&context, &particle_buffer, position_attrib_location);
    let zeroed = vec![0.0; MAX_PARTICLES * 2];
    unsafe {
        let positions_array_buf_view = js_sys::Float32Array::view(&zeroed);

        context.buffer_data_with_array_buffer_view(
            WebGlRenderingContext::ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGlRenderingContext::DYNAMIC_DRAW,
        );
    }

    Ok(RenderState {
        context,
        position_attrib_location,
        boundary_buffer,
        particle_buffer,
        draw_color_uniform,
    })
}

fn set_buffers_and_attributes(
    context: &WebGlRenderingContext,
    buffer: &WebGlBuffer,
    attrib_location: u32,
) {
    context.bind_buffer(WebGlRenderingContext::ARRAY_BUFFER, Some(buffer));
    context.vertex_attrib_pointer_with_i32(0, 2, WebGlRenderingContext::FLOAT, false, 0, 0);
    context.enable_vertex_attrib_array(attrib_location);
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
