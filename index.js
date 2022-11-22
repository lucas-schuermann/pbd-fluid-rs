import * as Stats from 'stats.js';
import * as dat from 'dat.gui';

import('./pkg').then(rust_wasm => {
    const $ = (id) => document.getElementById(id);

    // attach perf stats window
    const stats = new Stats();
    stats.dom.style.position = 'absolute';
    stats.showPanel(1); // ms per frame
    $('container').appendChild(stats.dom);

    // attach controls window
    const gui = new dat.GUI({ autoPlace: false });
    gui.domElement.style.opacity = 0.9;
    let props = {
        particles: 0,
        viscosity: 0,
        substeps: 10,
        block: () => {
            sim.add_block();
            setInfo();
        },
        reset: () => {
            sim.reset();
            setInfo();
        },
    };
    const setInfo = () => props.particles = sim.get_num_particles();
    gui.add(props, 'particles').listen();
    gui.add(props, 'viscosity', 0, 0.75, 0.01).onChange((v) => sim.set_viscosity(v));
    gui.add(props, 'substeps', 5, 10, 1).onChange((v) => sim.set_solver_substeps(v));
    gui.add(props, 'block').name("add block");
    gui.add(props, 'reset').name("reset simulation");
    $('gui').appendChild(gui.domElement);

    // import wasm package and initialize simulation
    const useDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const sim = new rust_wasm.Simulation(canvas, useDarkMode);
    setInfo();

    // main loop
    const step = () => {
        stats.begin(); // collect perf data for stats.js
        sim.step(); // update and redraw to canvas
        stats.end();
        requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}).catch(console.error);