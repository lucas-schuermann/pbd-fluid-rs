import * as Stats from 'stats.js';

import('./pkg').then(rust_wasm => {
    // attach perf stats window
    const stats = new Stats();
    stats.dom.style.position = 'absolute';
    stats.showPanel(1); // ms per frame
    document.getElementById('stats').appendChild(stats.dom);

    // import wasm package and initialize simulation
    const sim = new rust_wasm.Simulation(canvas);

    // bind interactivity
    const setInfo = () => document.getElementById('info').innerText = `Particles: ${sim.get_num_particles()}`;
    setInfo();
    document.getElementById('block').addEventListener('click', () => {
        sim.add_block();
        setInfo();
    });
    document.getElementById('reset').addEventListener('click', () => {
        sim.reset();
        setInfo();
    });

    // main loop
    const step = () => {
        stats.begin(); // collect perf data for stats.js
        sim.step(); // update and redraw to canvas
        stats.end();
        requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}).catch(console.error);