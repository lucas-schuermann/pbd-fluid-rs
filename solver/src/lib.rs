use cgmath::num_traits::{clamp, clamp_min};
use glam::{vec4, Vec2, Vec4};
use std::f32::consts::PI;

pub const G: Vec2 = glam::const_vec2!([0.0, -10.0]);
pub const WINDOW_WIDTH: u32 = 1280;
pub const WINDOW_HEIGHT: u32 = 800;

// TODO some drawing
pub const DRAW_ORIG: Vec2 =
    glam::const_vec2!([WINDOW_WIDTH as f32 / 2.0, WINDOW_HEIGHT as f32 - 20.0]);
pub const DRAW_SCALE: f32 = 200.0;

// TODO boundaries
pub const WIDTH: f32 = 1.0;
pub const HEIGHT: f32 = 2.0;

const PARTICLE_RADIUS: f32 = 0.01;
const UNILATERAL: bool = true;
const VISCOSITY: f32 = 0.0;
const TIME_STEP: f32 = 0.01;
const NUM_SOLVER_SUBSTEPS: usize = 10;
const DT: f32 = TIME_STEP / NUM_SOLVER_SUBSTEPS as f32;
const MAX_PARTICLES: usize = 10_000;

const PARTICLE_DIAMETER: f32 = 2.0 * PARTICLE_RADIUS;
const REST_DENSITY: f32 = 1.0 / (PARTICLE_DIAMETER * PARTICLE_DIAMETER);
const H: f32 = 3.0 * PARTICLE_RADIUS; // kernel radius
const H2: f32 = H * H;
const KERNEL_SCALE: f32 = 4.0 / (PI * H2 * H2 * H2 * H2); // 2d poly6 (SPH based shallow water simulation
const MAX_VEL: f32 = 0.4 * PARTICLE_RADIUS;

const HASH_SIZE: usize = 370111;
const GRID_SPACING: f32 = H * 1.5;
const G2: f32 = GRID_SPACING * GRID_SPACING;
const INV_GRID_SPACING: f32 = 1.0 / GRID_SPACING;

#[derive(Debug)]
pub struct Particles {
    pub pos: Vec<Vec2>,
    prev: Vec<Vec2>,
    vel: Vec<Vec2>,
}

#[derive(Debug)]
pub struct Grid {
    size: usize,
    first: Vec<Option<usize>>,
    marks: Vec<usize>,
    current_mark: usize,
    next: Vec<Option<usize>>,
    origin: Vec2,
}

impl Particles {
    pub fn new() -> Self {
        Self {
            pos: vec![Vec2::ZERO; MAX_PARTICLES],
            prev: vec![Vec2::ZERO; MAX_PARTICLES],
            vel: vec![Vec2::ZERO; MAX_PARTICLES],
        }
    }

    #[inline]
    pub fn positions(&self) -> &Vec<Vec2> {
        &self.pos
    }
}

impl Grid {
    pub fn new() -> Self {
        Self {
            size: HASH_SIZE,
            first: vec![None; HASH_SIZE],
            marks: vec![0; HASH_SIZE],
            current_mark: 0,
            next: vec![Some(0); MAX_PARTICLES],
            origin: Vec2::new(-100.0, -1.0),
        }
    }

    // TODO is this needed?
    pub fn clear(&mut self) {
        for i in 0..self.size {
            self.first[i] = None;
            self.marks[i] = 0;
        }
    }

    #[inline]
    pub fn map(&self, p: Vec2) -> (i32, i32) {
        let gx = f32::floor((p.x - self.origin.x) * INV_GRID_SPACING) as i32;
        let gy = f32::floor((p.y - self.origin.y) * INV_GRID_SPACING) as i32;
        (gx, gy)
    }

    #[inline]
    pub fn hash(&self, x: i32, y: i32) -> usize {
        (i32::abs(x.wrapping_mul(92837111i32) ^ y.wrapping_mul(689287499i32)) % (self.size as i32))
            as usize
    }
}

#[derive(Debug)]
pub struct State {
    pub particles: Particles,
    grads: Vec<Vec2>,
    pub num_particles: usize,
    boundaries: [Vec4; 2],
    grid: Grid,
    first_neighbor: Vec<usize>,
    neighbors: Vec<usize>,
}

impl State {
    pub fn new() -> Self {
        // specified as [left, right, bottom, top]
        let boundaries = [
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
        }
    }

    pub fn get_positions(&self) -> &[Vec2] {
        &self.particles.pos[0..self.num_particles]
    }

    pub fn clear(&mut self) {
        self.num_particles = 0;
        // TODO
    }

    pub fn init_dam_break(&mut self, dam_particles_x: usize, dam_particles_y: usize) {
        let fluid_orig = Vec2::new(-0.3, 1.8); // TODO (left, bottom)
        let eps = 0.00001;

        if dam_particles_x * dam_particles_y > MAX_PARTICLES {
            return; // TODO error
        }
        self.num_particles = dam_particles_x * dam_particles_y;
        let mut i = 0;
        for y in 0..dam_particles_y {
            for x in 0..dam_particles_x {
                self.particles.pos[i].x = fluid_orig.x + x as f32 * PARTICLE_DIAMETER;
                self.particles.pos[i].x += eps * (y % 2) as f32;
                self.particles.pos[i].y = fluid_orig.y + y as f32 * PARTICLE_DIAMETER;
                self.particles.vel[i] = Vec2::ZERO;
                i += 1;
            }
        }

        //self.grid.clear(); // TODO is this needed?
    }

    pub fn init_block(&mut self, _block_max_particles: usize) {}

    pub fn update(&mut self) {
        self.find_neighbors();

        for _ in 0..NUM_SOLVER_SUBSTEPS {
            // predict
            for i in 0..self.num_particles {
                self.particles.vel[i] += G * DT;
                self.particles.prev[i] = self.particles.pos[i];
                self.particles.pos[i] += self.particles.vel[i] * DT;
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
                self.particles.vel[i] = v / DT;
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
        const MIN_X: f32 = WINDOW_WIDTH as f32 * 0.5 / DRAW_SCALE; // TODO move
        for i in 0..self.num_particles {
            let p = &mut self.particles.pos[i];

            // ground
            p.y = clamp_min(p.y, 0.0);

            // left and right bounds
            p.x = clamp(p.x, -MIN_X, MIN_X);

            // boundary columns
            for b in self.boundaries {
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
                    self.particles.pos[id] += lambda * grad_i
                } else {
                    self.particles.pos[id] += lambda * self.grads[j];
                }
            }
        }
    }

    #[inline]
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
        self.particles.vel[i] += VISCOSITY * delta;
    }
}
