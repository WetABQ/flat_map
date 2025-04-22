use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use dashmap::DashMap;
use flat_table::Table;
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::sync::Arc;

fn bench_sequential_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_insert");
    
    for size in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Benchmark flat_table insertion
        group.bench_with_input(BenchmarkId::new("flat_table", size), size, |b, &size| {
            b.iter(|| {
                let mut table: Table<u64, u64> = Table::new_with_capacity(size);
                for i in 0..size {
                    table.insert(i as u64, i as u64);
                }
                table
            });
        });
        
        // Benchmark dashmap insertion
        group.bench_with_input(BenchmarkId::new("dashmap", size), size, |b, &size| {
            b.iter(|| {
                let table: DashMap<u64, u64> = DashMap::with_capacity(size);
                for i in 0..size {
                    table.insert(i as u64, i as u64);
                }
                table
            });
        });
    }
    
    group.finish();
}

fn bench_random_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_insert");
    let seed = 12345;
    
    for size in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        let mut rng = StdRng::seed_from_u64(seed);
        let keys: Vec<u64> = (0..*size).map(|_| rng.gen::<u64>()).collect();
        
        group.bench_with_input(BenchmarkId::new("flat_table", size), &keys, |b, keys| {
            b.iter(|| {
                let mut table: Table<u64, u64> = Table::new_with_capacity(keys.len());
                for &key in keys {
                    table.insert(key, key);
                }
                table
            });
        });
        
        group.bench_with_input(BenchmarkId::new("dashmap", size), &keys, |b, keys| {
            b.iter(|| {
                let table: DashMap<u64, u64> = DashMap::with_capacity(keys.len());
                for &key in keys {
                    table.insert(key, key);
                }
                table
            });
        });
    }
    
    group.finish();
}

fn bench_sequential_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("sequential_get");
    
    for size in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Prepare data
        let mut flat_table: Table<u64, u64> = Table::new_with_capacity(*size);
        let dashmap: DashMap<u64, u64> = DashMap::with_capacity(*size);
        
        for i in 0..*size {
            flat_table.insert(i as u64, i as u64);
            dashmap.insert(i as u64, i as u64);
        }
        
        // Benchmark flat_table reads
        group.bench_with_input(BenchmarkId::new("flat_table", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(flat_table.get(&(i as u64)));
                }
            });
        });
        
        // Benchmark dashmap reads
        group.bench_with_input(BenchmarkId::new("dashmap", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    black_box(dashmap.get(&(i as u64)));
                }
            });
        });
    }
    
    group.finish();
}

fn bench_random_get(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_get");
    let seed = 67890;
    
    for size in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Prepare data with sequential keys
        let mut flat_table: Table<u64, u64> = Table::new_with_capacity(*size);
        let dashmap: DashMap<u64, u64> = DashMap::with_capacity(*size);
        
        for i in 0..*size {
            flat_table.insert(i as u64, i as u64);
            dashmap.insert(i as u64, i as u64);
        }
        
        // Generate random access pattern
        let mut rng = StdRng::seed_from_u64(seed);
        let access_pattern: Vec<u64> = (0..*size).map(|_| rng.gen_range(0..*size) as u64).collect();
        
        // Benchmark flat_table random reads
        group.bench_with_input(BenchmarkId::new("flat_table", size), &access_pattern, |b, pattern| {
            b.iter(|| {
                for &key in pattern {
                    black_box(flat_table.get(&key));
                }
            });
        });
        
        // Benchmark dashmap random reads
        group.bench_with_input(BenchmarkId::new("dashmap", size), &access_pattern, |b, pattern| {
            b.iter(|| {
                for &key in pattern {
                    black_box(dashmap.get(&key));
                }
            });
        });
    }
    
    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");
    
    for size in [100_000, 1_000_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Prepare data
        let mut flat_table: Table<u64, u64> = Table::new_with_capacity(*size);
        let dashmap: DashMap<u64, u64> = DashMap::with_capacity(*size);
        
        for i in 0..*size {
            flat_table.insert(i as u64, 0);
            dashmap.insert(i as u64, 0);
        }
        
        // Benchmark flat_table updates (reinsert with new value)
        group.bench_with_input(BenchmarkId::new("flat_table", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    flat_table.insert(i as u64, i as u64);
                }
            });
        });
        
        // Benchmark dashmap updates
        group.bench_with_input(BenchmarkId::new("dashmap", size), size, |b, &size| {
            b.iter(|| {
                for i in 0..size {
                    dashmap.insert(i as u64, i as u64);
                }
            });
        });
    }
    
    group.finish();
}

fn bench_mixed_workload(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_workload");
    let seed = 42;
    
    for size in [100_000].iter() {
        group.throughput(Throughput::Elements(*size as u64));
        
        // Create initial tables with half the capacity filled
        let mut flat_table: Table<u64, u64> = Table::new_with_capacity(*size);
        let dashmap: Arc<DashMap<u64, u64>> = Arc::new(DashMap::with_capacity(*size));
        
        let half_size = *size / 2;
        for i in 0..half_size {
            flat_table.insert(i as u64, i as u64);
            dashmap.insert(i as u64, i as u64);
        }
        
        // Generate random operations (33% insert, 33% get, 33% update)
        let mut rng = StdRng::seed_from_u64(seed);
        let operations: Vec<(u8, u64)> = (0..*size)
            .map(|_| {
                let op_type = rng.gen_range(0..3);  // 0=insert, 1=get, 2=update
                let key = if op_type == 0 {
                    // For inserts, use keys beyond what's initially inserted
                    rng.gen_range(half_size..*size) as u64
                } else {
                    // For gets and updates, use keys from the initial set
                    rng.gen_range(0..half_size) as u64
                };
                (op_type, key)
            })
            .collect();
        
        // Benchmark flat_table mixed workload
        group.bench_with_input(BenchmarkId::new("flat_table", size), &operations, |b, ops| {
            b.iter(|| {
                let mut table = flat_table.clone();
                for &(op_type, key) in ops {
                    match op_type {
                        0 => table.insert(key, key),  // Insert
                        1 => { black_box(table.get(&key)); }  // Get
                        2 => table.insert(key, key + 1),  // Update
                        _ => unreachable!(),
                    }
                }
            });
        });
        
        // Benchmark dashmap mixed workload
        group.bench_with_input(BenchmarkId::new("dashmap", size), &operations, |b, ops| {
            b.iter(|| {
                let table = dashmap.clone();
                for &(op_type, key) in ops {
                    match op_type {
                        0 => { table.insert(key, key); }  // Insert
                        1 => { black_box(table.get(&key)); }  // Get
                        2 => { table.insert(key, key + 1); }  // Update
                        _ => unreachable!(),
                    }
                }
            });
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    bench_sequential_insert,
    bench_random_insert,
    bench_sequential_get,
    bench_random_get,
    bench_update,
    bench_mixed_workload
);
criterion_main!(benches);
