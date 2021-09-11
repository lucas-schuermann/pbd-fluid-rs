import * as Stats from 'stats.js';

import('./pkg').then(rust_wasm => {
    // check if required features are supported, else show error and exit
    if (!document.getElementById('canvas').getContext('webgl2')) {
        const errorString = 'Browser does not support required features';
        console.error(errorString);
        const canvas = document.getElementById('canvas');
        canvas.style = 'border: 1px solid red';
        const ctx = canvas.getContext('2d');
        ctx.font = '16px serif';
        ctx.fillText(errorString, 10, 25);
        return;
    }

    // attach perf stats window
    const stats = new Stats();
    stats.dom.style.position = 'absolute';
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