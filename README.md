# pbd-fluid-rs
2D Position Based Fluid implementation in Rust with WASM + WebGL

Inspired by Matthais Müller-Fischer's [position based fluid demo](https://matthias-research.github.io/pages/challenges/fluid2d.html)

## Usage
Install dependencies
```
npm install
rustup install nightly
```

Compile WASM, run webpack, and spawn a local server (note you might need to download additional rust-src, etc. with rustup)
```
npm run serve
```
Then visit `http://localhost:8080`
