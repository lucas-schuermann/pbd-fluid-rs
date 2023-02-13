use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn dam_break(n: usize, i: usize) {
    let mut state = solver::State::new(2.0);
    state.init_dam_break(10, n / 10);
    for _ in 0..i {
        state.update();
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("sample-size-10");
    group.sample_size(10);
    group.warm_up_time(std::time::Duration::from_secs(20));
    group.bench_function("dam break: n=5000, i=100", |b| {
        b.iter(|| dam_break(black_box(5000), black_box(100)));
    });
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
