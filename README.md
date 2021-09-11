# pbd-fluid-rs
2D Position Based Fluid implementation in Rust with WASM + WebGL

Inspired by Matthais Müller's [position based fluid demo](https://matthias-research.github.io/pages/challenges/fluid2d.html)

## Usage
### Native
Run with cargo:
```
RUST_LOG=info cargo r --package native --release
```
Press `r` to reset simulation or `space` to add a block of particles

### Web
Install dependencies
```
npm install
```

Compile WASM, run webpack, and spawn a local server
```
npm run serve
```
Then visit `http://localhost:8080`
