use criterion::{criterion_group, criterion_main, Criterion};
use std::hint::black_box;

fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 1,
        1 => 1,
        n => fibonacci(n - 1) + fibonacci(n - 2),
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("slow brain", |b| b.iter(|| revol::init_brain(black_box(4))));

    c.bench_function("fast brain", |b| {
        b.iter(|| revol::brain::default_init_brain(black_box(4)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
