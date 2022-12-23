![Build](https://github.com/cerrno/pbd-fluid-rs/actions/workflows/main.yml/badge.svg)

`pbd-fluid-rs` is a 2D Position Based Fluid implementation in Rust with WASM + WebGL. It was inspired by Matthais Müller's [position based fluid demo](https://matthias-research.github.io/pages/challenges/fluid2d.html).

For a quick demo, please see https://cerrno.github.io/pbd-fluid-rs/. The WASM version of this project is deployed to Github Pages after building with Github Actions.

## Running
### Package Dependencies
```bash
# debian/ubuntu
apt install build-essential pkg-config cmake libfreetype6-dev libfontconfig1-dev
```

### Native (cargo)
```bash
RUST_LOG=info cargo run --package native --release
```
Press `r` to reset simulation or `space` to add a block of particles

### Web (npm)
```bash
# install dependencies
npm install

# compile to WASM, run webpack, and spawn a local server
npm run serve
```
Then visit http://localhost:8080

## License
This project is distributed under the [MIT license](LICENSE.md).