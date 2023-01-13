import * as Stats from 'stats.js';
import GUI from 'lil-gui';

import('./pkg').then(rust_wasm => {
    const $ = (id: string) => document.getElementById(id);

    // attach perf stats window
    const stats = new Stats();
    stats.dom.style.position = 'absolute';
    const simPanel = stats.addPanel(new Stats.Panel('MS (Sim)', '#ff8', '#221'));
    let maxSimMs = 1;
    stats.showPanel(3); // ms per sim step
    $('container').appendChild(stats.dom);

    // attach controls window
    const gui = new GUI({ autoPlace: false });
    gui.domElement.style.opacity = '0.9';
    let props = {
        particles: 0,
        viscosity: 0,
        substeps: 10,
        singleColor: false,
        block: () => {
            sim.add_block();
            setInfo();
        },
        reset10x1000: () => reset(10, 1000),
        reset10x200: () => reset(10, 200),
        reset40x100: () => reset(40, 100),
        reset100x100: () => reset(100, 100),
    };
    const reset = (x: number, y: number) => {
        sim.reset(x, y);
        setInfo();
    }
    const setInfo = () => {
        props.particles = sim.num_particles;
        particlesControl.updateDisplay();
    };
    const particlesControl = gui.add(props, 'particles').disable();
    gui.add(props, 'viscosity', 0, 0.75, 0.005).onChange((v: number) => sim.viscosity = v);
    gui.add(props, 'substeps', 5, 10, 1).onChange((v: number) => sim.solver_substeps = v);
    gui.add(props, 'singleColor').name('draw single color').onFinishChange((v: boolean) => sim.draw_single_color = v);
    gui.add(props, 'block').name('add block');
    gui.add(props, 'reset10x1000').name('reset 10x1000 block');
    gui.add(props, 'reset10x200').name('reset 10x200 block');
    gui.add(props, 'reset40x100').name('reset 40x100 block');
    gui.add(props, 'reset100x100').name('reset 100x100 block');
    $('gui').appendChild(gui.domElement);

    // import wasm package and initialize simulation
    const useDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
    const sim = new rust_wasm.Simulation($('canvas') as HTMLCanvasElement, useDarkMode);
    setInfo();

    // main loop
    const animate = () => {
        stats.begin(); // collect perf data for stats.js
        let simTimeMs = performance.now();
        sim.step(); // update and redraw to canvas
        simTimeMs = performance.now() - simTimeMs;
        sim.draw();
        simPanel.update(simTimeMs, (maxSimMs = simTimeMs > maxSimMs ? simTimeMs : maxSimMs) * 2);
        stats.end();
        requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
}).catch(console.error);