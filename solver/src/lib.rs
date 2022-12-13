use std::f32::consts::PI;

use cgmath::num_traits::{clamp, clamp_min};
use glam::{vec2, vec4, Vec2, Vec4};

pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 600;

pub const DRAW_ORIG: Vec2 = vec2(WINDOW_WIDTH as f32 / 2.0, WINDOW_HEIGHT as f32);
pub const DRAW_SCALE: f32 = 200.0;

// boundaries
pub const WIDTH: f32 = 1.0;
pub const HEIGHT: f32 = 2.0;
const MIN_X: f32 = WINDOW_WIDTH as f32 * 0.5 / DRAW_SCALE;

pub const MAX_PARTICLES: usize = 20_000;
const G: Vec2 = vec2(0.0, -10.0);
const PARTICLE_RADIUS: f32 = 0.01;
const UNILATERAL: bool = true;
const DEFAULT_VISCOSITY: f32 = 0.0;
const TIME_STEP: f32 = 0.01;
const DEFAULT_NUM_SOLVER_SUBSTEPS: usize = 10;
const PARTICLE_DIAMETER: f32 = 2.0 * PARTICLE_RADIUS;
const REST_DENSITY: f32 = 1.0 / (PARTICLE_DIAMETER * PARTICLE_DIAMETER);
const H: f32 = 3.0 * PARTICLE_RADIUS; // kernel radius
const H2: f32 = H * H;
const KERNEL_SCALE: f32 = 4.0 / (PI * H2 * H2 * H2 * H2); // 2d poly6 (SPH based shallow water simulation)
const MAX_VEL: f32 = 0.4 * PARTICLE_RADIUS;

const HASH_SIZE: usize = 370_111;
const GRID_SPACING: f32 = H * 1.5;
const G2: f32 = GRID_SPACING * GRID_SPACING;
const INV_GRID_SPACING: f32 = 1.0 / GRID_SPACING;

#[derive(Debug)]
struct Particles {
    pos: Vec<Vec2>,
    prev: Vec<Vec2>,
    vel: Vec<Vec2>,
}

#[derive(Debug)]
struct Grid {
    size: usize,
    first: Vec<Option<usize>>,
    marks: Vec<usize>,
    current_mark: usize,
    next: Vec<Option<usize>>,
    origin: Vec2,
}

impl Particles {
    fn new() -> Self {
        Self {
            pos: vec![Vec2::ZERO; MAX_PARTICLES],
            prev: vec![Vec2::ZERO; MAX_PARTICLES],
            vel: vec![Vec2::ZERO; MAX_PARTICLES],
        }
    }
}

impl Grid {
    fn new() -> Self {
        Self {
            size: HASH_SIZE,
            first: vec![None; HASH_SIZE],
            marks: vec![0; HASH_SIZE],
            current_mark: 0,
            next: vec![Some(0); MAX_PARTICLES],
            origin: Vec2::new(-100.0, -1.0),
        }
    }

    fn map(&self, p: Vec2) -> (i32, i32) {
        let gx = f32::floor((p.x - self.origin.x) * INV_GRID_SPACING) as i32;
        let gy = f32::floor((p.y - self.origin.y) * INV_GRID_SPACING) as i32;
        (gx, gy)
    }

    const fn hash(&self, x: i32, y: i32) -> usize {
        (i32::abs(x.wrapping_mul(92_837_111_i32) ^ y.wrapping_mul(689_287_499_i32))
            % (self.size as i32)) as usize
    }
}

#[derive(Debug)]
pub struct State {
    particles: Particles,
    grads: Vec<Vec2>,
    pub num_particles: usize,
    boundaries: Vec<Vec4>,
    grid: Grid,
    first_neighbor: Vec<usize>,
    neighbors: Vec<usize>,

    viscosity: f32,
    num_substeps: usize,
    dt: f32,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    #[must_use]
    pub fn new() -> Self {
        // specified as [x0, x0+width, y0, y0+height]
        let boundaries = vec![
            vec4(-WIDTH * 0.5 - 0.1, -WIDTH * 0.5, -0.01, HEIGHT), // left column
            vec4(WIDTH * 0.5, WIDTH * 0.5 + 0.1, -0.01, HEIGHT),   // right column
        ];
        Self {
            particles: Particles::new(),
            grads: vec![Vec2::ZERO; 1000],
            num_particles: 0,
            boundaries,
            grid: Grid::new(),
            first_neighbor: vec![0; MAX_PARTICLES + 1],
            neighbors: Vec::with_capacity(MAX_PARTICLES * 10),
            viscosity: DEFAULT_VISCOSITY,
            num_substeps: DEFAULT_NUM_SOLVER_SUBSTEPS,
            dt: TIME_STEP / DEFAULT_NUM_SOLVER_SUBSTEPS as f32,
        }
    }

    #[must_use]
    pub fn get_boundaries(&self) -> Vec<[f32; 4]> {
        self.boundaries.iter().map(Vec4::to_array).collect()
    }

    #[must_use]
    pub fn get_positions(&self) -> &[Vec2] {
        &self.particles.pos[0..self.num_particles]
    }

    pub fn set_viscosity(&mut self, viscosity: f32) {
        self.viscosity = viscosity;
    }

    pub fn set_solver_substeps(&mut self, num_substeps: usize) {
        self.num_substeps = num_substeps;
        self.dt = TIME_STEP / num_substeps as f32;
    }

    pub fn clear(&mut self) {
        for i in 0..self.num_particles {
            self.particles.pos[i] = Vec2::ZERO;
            self.particles.prev[i] = Vec2::ZERO;
            self.particles.vel[i] = Vec2::ZERO;
        }
        self.num_particles = 0;
    }

    /// Returns `false` if new particles would overflow `MAX_PARTICLES`
    pub fn init_dam_break(&mut self, dam_particles_x: usize, dam_particles_y: usize) -> bool {
        let fluid_orig = Vec2::new(-0.3, 1.8); // (left, bottom)
        let eps = 0.00001;

        if self.num_particles + dam_particles_x * dam_particles_y > MAX_PARTICLES {
            return false;
        }

        let mut i = self.num_particles;
        for y in 0..dam_particles_y {
            for x in 0..dam_particles_x {
                self.particles.pos[i].x = fluid_orig.x + x as f32 * PARTICLE_DIAMETER;
                self.particles.pos[i].x += eps * (y % 2) as f32;
                self.particles.pos[i].y = fluid_orig.y + y as f32 * PARTICLE_DIAMETER;
                self.particles.vel[i] = Vec2::ZERO;
                i += 1;
            }
        }
        self.num_particles += dam_particles_x * dam_particles_y;
        true
    }

    /// Returns `false` if new particles would overflow `MAX_PARTICLES`
    pub fn init_block(&mut self, block_max_particles: usize) -> bool {
        let fluid_orig = Vec2::new(0.0, 2.5); // (left, bottom)
        let eps = 0.00001;

        if self.num_particles + block_max_particles > MAX_PARTICLES {
            return false;
        }

        let bound = f32::sqrt(block_max_particles as f32) as usize;
        let mut i = self.num_particles;
        for y in 0..bound {
            for x in 0..bound {
                self.particles.pos[i].x = fluid_orig.x + x as f32 * PARTICLE_DIAMETER;
                self.particles.pos[i].x += eps * (y % 2) as f32;
                self.particles.pos[i].y = fluid_orig.y + y as f32 * PARTICLE_DIAMETER;
                self.particles.vel[i] = Vec2::ZERO;
                i += 1;
            }
        }
        self.num_particles += block_max_particles;
        true
    }

    pub fn update(&mut self) {
        self.find_neighbors();

        for _ in 0..self.num_substeps {
            // predict
            for i in 0..self.num_particles {
                self.particles.vel[i] += G * self.dt;
                self.particles.prev[i] = self.particles.pos[i];
                self.particles.pos[i] += self.particles.vel[i] * self.dt;
            }

            // solve
            self.solve_boundaries();
            self.solve_fluid();

            // derive velocities
            for i in 0..self.num_particles {
                let mut v = self.particles.pos[i] - self.particles.prev[i];
                let vel = v.length();
                // CFL
                if vel > MAX_VEL {
                    v *= MAX_VEL / vel;
                    self.particles.pos[i] = self.particles.prev[i] + v;
                }
                self.particles.vel[i] = v / self.dt;
                self.apply_viscosity(i);
            }
        }
    }

    fn find_neighbors(&mut self) {
        // hash particles into grid
        self.grid.current_mark += 1;
        for i in 0..self.num_particles {
            let (gx, gy) = self.grid.map(self.particles.pos[i]);
            let h = self.grid.hash(gx, gy);
            if self.grid.marks[h] != self.grid.current_mark {
                self.grid.marks[h] = self.grid.current_mark;
                self.grid.first[h] = None;
            }
            self.grid.next[i] = self.grid.first[h];
            self.grid.first[h] = Some(i);
        }

        // collect neighbors
        self.neighbors.clear();
        for i in 0..self.num_particles {
            let p = self.particles.pos[i];
            self.first_neighbor[i] = self.neighbors.len();
            let (gx, gy) = self.grid.map(p);
            for x in gx - 1..=(gx + 1) {
                for y in gy - 1..=(gy + 1) {
                    let h = self.grid.hash(x, y);
                    if self.grid.marks[h] != self.grid.current_mark {
                        continue;
                    }
                    let mut j = self.grid.first[h];
                    while let Some(ji) = j {
                        let d = self.particles.pos[ji] - p;
                        if d.length_squared() < G2 {
                            self.neighbors.push(ji);
                        }
                        j = self.grid.next[ji];
                    }
                }
            }
        }
        self.first_neighbor[self.num_particles] = self.neighbors.len();
    }

    fn solve_boundaries(&mut self) {
        for i in 0..self.num_particles {
            let p = &mut self.particles.pos[i];

            // ground
            p.y = clamp_min(p.y, 0.0);

            // left and right bounds
            p.x = clamp(p.x, -MIN_X, MIN_X);

            // boundary columns
            for b in &self.boundaries {
                if p.x < b.x || p.x > b.y || p.y < b.z || p.y > b.w {
                    continue;
                }

                let mut d = Vec2::ZERO;
                if p.x < (b.x + b.y) * 0.5 {
                    d.x = b.x - p.x;
                } else {
                    d.x = b.y - p.x;
                }
                if p.y < (b.z + b.w) * 0.5 {
                    d.y = b.z - p.y;
                } else {
                    d.y = b.w - p.y;
                }

                if d.x.abs() < d.y.abs() {
                    p.x += d.x;
                } else {
                    p.y += d.y;
                }
            }
        }
    }

    fn solve_fluid(&mut self) {
        for i in 0..self.num_particles {
            let p = self.particles.pos[i];
            let first = self.first_neighbor[i];
            let num_neighbors = self.first_neighbor[i + 1] - first;

            let mut rho = 0.0;
            let mut sum_grad2 = 0.0;
            let mut grad_i = Vec2::ZERO;
            for j in 0..num_neighbors {
                let id = self.neighbors[first + j];
                let mut n = self.particles.pos[id] - p;
                let r = n.length();
                // normalize
                if r > 0.0 {
                    n /= r;
                }
                if r > H {
                    self.grads[j] = Vec2::ZERO;
                } else {
                    let r2 = r * r;
                    let w = H2 - r2;
                    rho += KERNEL_SCALE * w * w * w;
                    let grad = (KERNEL_SCALE * 3.0 * w * w * (-2.0 * r)) / REST_DENSITY;
                    self.grads[j] = n * grad;
                    grad_i -= n * grad;
                    sum_grad2 += grad * grad;
                }
            }

            sum_grad2 += grad_i.length_squared();
            let c = rho / REST_DENSITY - 1.0;
            if UNILATERAL && c < 0.0 {
                continue;
            }

            let lambda = -c / (sum_grad2 + 0.0001);
            for j in 0..num_neighbors {
                let id = self.neighbors[first + j];
                if id == i {
                    self.particles.pos[id] += lambda * grad_i;
                } else {
                    self.particles.pos[id] += lambda * self.grads[j];
                }
            }
        }
    }

    fn apply_viscosity(&mut self, i: usize) {
        let first = self.first_neighbor[i];
        let num_neighbors = self.first_neighbor[i + 1] - first;
        if num_neighbors == 0 {
            return;
        }
        let mut avg_vel = Vec2::ZERO;
        for j in 0..num_neighbors {
            let id = self.neighbors[first + j];
            avg_vel += self.particles.vel[id];
        }
        avg_vel /= num_neighbors as f32;
        let delta = avg_vel - self.particles.vel[i];
        self.particles.vel[i] += self.viscosity * delta;
    }
}
