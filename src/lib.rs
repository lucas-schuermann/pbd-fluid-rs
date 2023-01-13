#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    WebGl2RenderingContext, WebGlBuffer, WebGlProgram, WebGlShader, WebGlUniformLocation,
};

// defaults for init scene
const DAM_PARTICLES_X: usize = 10;
const DAM_PARTICLES_Y: usize = 1000;

const BLOCK_PARTICLES: usize = 400;
const MAX_PARTICLES: usize = solver::MAX_PARTICLES;
const POINT_SIZE: f32 = 3.0;

#[wasm_bindgen]
pub struct Simulation {
    renderer: RenderState,
    state: solver::State,
}

struct RenderState {
    context: WebGl2RenderingContext,
    boundary_buffer: WebGlBuffer,
    particle_buffer: WebGlBuffer,
    draw_mode_single_color_uniform: WebGlUniformLocation,
    draw_mode_boundary_uniform: WebGlUniformLocation,
    position_attrib_location: u32,
}

#[wasm_bindgen]
impl Simulation {
    /// # Errors
    /// Will return `Err` if unable to initialize webgl2 context and compile/link shader programs.
    #[wasm_bindgen(constructor)]
    pub fn new(
        canvas: &web_sys::HtmlCanvasElement,
        use_dark_colors: bool,
    ) -> Result<Simulation, JsValue> {
        let mut state = solver::State::new();
        state.init_dam_break(DAM_PARTICLES_X, DAM_PARTICLES_Y);
        let renderer = init_webgl(canvas, &state.get_boundaries(), use_dark_colors)?;
        Ok(Simulation { renderer, state })
    }

    #[wasm_bindgen(setter)]
    pub fn set_draw_single_color(&self, enabled: bool) {
        self.renderer.context.uniform1i(
            Some(&self.renderer.draw_mode_single_color_uniform),
            enabled.into(),
        );
    }

    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn num_particles(&self) -> usize {
        self.state.num_particles
    }

    #[wasm_bindgen(setter)]
    pub fn set_viscosity(&mut self, viscosity: f32) {
        self.state.set_viscosity(viscosity);
    }

    #[wasm_bindgen(setter)]
    pub fn set_solver_substeps(&mut self, num_substeps: usize) {
        self.state.set_solver_substeps(num_substeps);
    }

    pub fn step(&mut self) {
        self.state.update();
    }

    pub fn add_block(&mut self) {
        self.state.init_block(BLOCK_PARTICLES);
    }

    pub fn reset(&mut self, dam_particles_x: usize, dam_particles_y: usize) {
        self.state.clear();
        self.state.init_dam_break(dam_particles_x, dam_particles_y);
    }

    pub fn draw(&self) {
        self.renderer
            .context
            .clear(WebGl2RenderingContext::COLOR_BUFFER_BIT);

        // draw boundaries
        set_buffers_and_attributes(
            &self.renderer.context,
            &self.renderer.boundary_buffer,
            self.renderer.position_attrib_location,
        );
        self.renderer
            .context
            .uniform1i(Some(&self.renderer.draw_mode_boundary_uniform), 1);
        self.renderer
            .context
            .draw_arrays(WebGl2RenderingContext::TRIANGLES, 0, 12);

        // draw particles
        set_buffers_and_attributes(
            &self.renderer.context,
            &self.renderer.particle_buffer,
            self.renderer.position_attrib_location,
        );
        unsafe {
            // Note that `Float32Array::view` is somewhat dangerous (hence the
            // `unsafe`!). This is creating a raw view into our module's
            // `WebAssembly.Memory` buffer, but if we allocate more pages for ourself
            // (aka do a memory allocation in Rust) it'll cause the buffer to change,
            // causing the `Float32Array` to be invalid.
            //
            // As a result, after `Float32Array::view` we have to be very careful not to
            // do any memory allocations before it's dropped.
            let positions_f32_view = self.state.get_positions().as_ptr().cast::<f32>(); // &[Vec2] -> *const Vec2 -> *const f32
            let positions_array_buf_view = js_sys::Float32Array::view(std::slice::from_raw_parts(
                positions_f32_view,
                self.state.num_particles * 2,
            ));

            self.renderer
                .context
                .buffer_sub_data_with_i32_and_array_buffer_view(
                    WebGl2RenderingContext::ARRAY_BUFFER,
                    0,
                    &positions_array_buf_view,
                );
        }
        self.renderer
            .context
            .uniform1i(Some(&self.renderer.draw_mode_boundary_uniform), 0);
        self.renderer.context.draw_arrays(
            WebGl2RenderingContext::POINTS,
            0,
            self.state.num_particles as i32,
        );
    }
}

#[allow(clippy::too_many_lines)]
fn init_webgl(
    canvas: &web_sys::HtmlCanvasElement,
    boundaries: &[[f32; 4]],
    use_dark_colors: bool,
) -> Result<RenderState, JsValue> {
    // set up canvas and webgl context handle
    canvas.set_width(solver::WINDOW_WIDTH);
    canvas.set_height(solver::WINDOW_HEIGHT);
    let context = canvas
        .get_context("webgl2")?
        .unwrap()
        .dyn_into::<WebGl2RenderingContext>()?;

    context.viewport(
        0,
        0,
        solver::WINDOW_WIDTH as i32,
        solver::WINDOW_HEIGHT as i32,
    );
    if use_dark_colors {
        context.clear_color(0.1, 0.1, 0.1, 1.0);
    } else {
        context.clear_color(0.9, 0.9, 0.9, 1.0);
    }

    let vert_shader = compile_shader(
        &context,
        WebGl2RenderingContext::VERTEX_SHADER,
        format!(
            r##"#version 300 es
        precision mediump float;
        const vec4 particle_color_1 = vec4(0.2549019608, 0.4117647059, 1.0, 1.0); // #4169E1
        const vec4 particle_color_2 = vec4(1.0, 0.2549019608, 0.2980392157, 1.0); // #E1414C

        uniform mat4 projection_matrix;
        uniform mat4 view_matrix;
        uniform int draw_mode_single_color;
        in vec2 position;
        out vec4 color;

        void main() {{
            gl_PointSize = {:.1};
            gl_Position = projection_matrix * view_matrix * vec4(position, 0.0, 1.0);
            if (draw_mode_single_color == 1 || int(floor(float(gl_VertexID) / 1000.0)) % 2 == 0) {{
                color = particle_color_1;
            }} else {{
                color = particle_color_2;
            }}
        }}
        "##,
            POINT_SIZE
        )
        .as_str(),
    )?;

    let frag_shader = compile_shader(
        &context,
        WebGl2RenderingContext::FRAGMENT_SHADER,
        r#"#version 300 es
        precision mediump float;
        const vec4 boundary_color = vec4(0.4392156863, 0.5019607843, 0.5647058824, 1.0); // #708090

        uniform int draw_mode_boundary;
        in vec4 color;
        out vec4 f_color;

        void main() {
            if (draw_mode_boundary == 1) {
                f_color = boundary_color;
            } else {
                f_color = color;
            }
        }
        "#,
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
    let draw_mode_single_color_uniform = context
        .get_uniform_location(&program, "draw_mode_single_color")
        .expect("Unable to get vertex color mode uniform location");
    context.uniform1i(Some(&draw_mode_single_color_uniform), 0);
    let draw_mode_boundary_uniform = context
        .get_uniform_location(&program, "draw_mode_boundary")
        .expect("Unable to get fragment boundary uniform location");
    let position_attrib_location = context.get_attrib_location(&program, "position") as u32;

    // prepopulate boundary geometry
    let boundaries: Vec<f32> = boundaries
        .iter()
        .flat_map(|p| {
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
        .collect();
    let boundary_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    set_buffers_and_attributes(&context, &boundary_buffer, position_attrib_location);
    unsafe {
        let boundaries_array_buf_view = js_sys::Float32Array::view(&boundaries);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &boundaries_array_buf_view,
            WebGl2RenderingContext::STATIC_DRAW,
        );
    }

    // preallocate particle vertex buffer
    let particle_buffer = context.create_buffer().ok_or("Failed to create buffer")?;
    set_buffers_and_attributes(&context, &particle_buffer, position_attrib_location);
    let zeroed = vec![0.0; MAX_PARTICLES * 2];
    unsafe {
        let positions_array_buf_view = js_sys::Float32Array::view(&zeroed);

        context.buffer_data_with_array_buffer_view(
            WebGl2RenderingContext::ARRAY_BUFFER,
            &positions_array_buf_view,
            WebGl2RenderingContext::DYNAMIC_DRAW,
        );
    }

    Ok(RenderState {
        context,
        boundary_buffer,
        particle_buffer,
        draw_mode_single_color_uniform,
        draw_mode_boundary_uniform,
        position_attrib_location,
    })
}

fn set_buffers_and_attributes(
    context: &WebGl2RenderingContext,
    buffer: &WebGlBuffer,
    attrib_location: u32,
) {
    context.bind_buffer(WebGl2RenderingContext::ARRAY_BUFFER, Some(buffer));
    context.vertex_attrib_pointer_with_i32(0, 2, WebGl2RenderingContext::FLOAT, false, 0, 0);
    context.enable_vertex_attrib_array(attrib_location);
}

fn compile_shader(
    context: &WebGl2RenderingContext,
    shader_type: u32,
    source: &str,
) -> Result<WebGlShader, String> {
    let shader = context
        .create_shader(shader_type)
        .ok_or_else(|| String::from("Unable to create shader object"))?;
    context.shader_source(&shader, source);
    context.compile_shader(&shader);

    if context
        .get_shader_parameter(&shader, WebGl2RenderingContext::COMPILE_STATUS)
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
    context: &WebGl2RenderingContext,
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
        .get_program_parameter(&program, WebGl2RenderingContext::LINK_STATUS)
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
