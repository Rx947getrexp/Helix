use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::collections::HashMap;
use veclite::{HNSWConfig, IndexType, Metadata, VecLite, VecLiteConfig, VectorData};

fn create_test_vectors(count: usize, dimensions: usize) -> Vec<(String, VectorData, Metadata)> {
    (0..count)
        .map(|i| {
            let id = format!("vec_{}", i);
            let vector: VectorData = (0..dimensions).map(|j| ((i + j) as f32) / 100.0).collect();
            let metadata = HashMap::from([
                (
                    "type".to_string(),
                    if i % 2 == 0 {
                        "even".to_string()
                    } else {
                        "odd".to_string()
                    },
                ),
                ("index".to_string(), i.to_string()),
            ]);
            (id, vector, metadata)
        })
        .collect()
}

fn bench_vector_insertion(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_insertion");

    for vector_count in [100, 1000, 10000].iter() {
        for dimensions in [64, 256, 768].iter() {
            let vectors = create_test_vectors(*vector_count, *dimensions);

            group.throughput(Throughput::Elements(*vector_count as u64));
            group.bench_with_input(
                BenchmarkId::new("single_insert", format!("{}x{}", vector_count, dimensions)),
                &vectors,
                |b, vectors| {
                    b.iter_with_setup(
                        || VecLite::new().unwrap(),
                        |db| {
                            for (id, vector, metadata) in vectors.iter() {
                                black_box(
                                    db.insert(id.clone(), vector.clone(), metadata.clone())
                                        .unwrap(),
                                );
                            }
                        },
                    );
                },
            );

            group.bench_with_input(
                BenchmarkId::new("batch_insert", format!("{}x{}", vector_count, dimensions)),
                &vectors,
                |b, vectors| {
                    b.iter_with_setup(
                        || VecLite::new().unwrap(),
                        |db| {
                            black_box(db.insert_batch(vectors.clone()).unwrap());
                        },
                    );
                },
            );
        }
    }

    group.finish();
}

fn bench_vector_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_search");

    for vector_count in [1000, 10000].iter() {
        for dimensions in [64, 256, 768].iter() {
            let vectors = create_test_vectors(*vector_count, *dimensions);
            let db = VecLite::new().unwrap();
            db.insert_batch(vectors).unwrap();

            let query: VectorData = (0..*dimensions).map(|i| (i as f32) / 100.0).collect();

            for k in [1, 10, 100].iter() {
                group.throughput(Throughput::Elements(*vector_count as u64));
                group.bench_with_input(
                    BenchmarkId::new(
                        "euclidean_search",
                        format!("{}x{}_k{}", vector_count, dimensions, k),
                    ),
                    &(query.clone(), *k),
                    |b, (q, k)| b.iter(|| black_box(db.search(q, *k).unwrap())),
                );

                group.bench_with_input(
                    BenchmarkId::new(
                        "cosine_search",
                        format!("{}x{}_k{}", vector_count, dimensions, k),
                    ),
                    &(query.clone(), *k),
                    |b, (q, k)| {
                        b.iter(|| black_box(db.search_with_metric(q, *k, "cosine").unwrap()))
                    },
                );
            }
        }
    }

    group.finish();
}

fn bench_distance_metrics(c: &mut Criterion) {
    use veclite::distance::{
        CosineDistance, DistanceMetric, DotProductSimilarity, EuclideanDistance,
    };

    let mut group = c.benchmark_group("distance_metrics");

    for dimensions in [64, 256, 768, 1024].iter() {
        let a: VectorData = (0..*dimensions).map(|i| (i as f32) / 100.0).collect();
        let b: VectorData = (0..*dimensions).map(|i| ((i + 1) as f32) / 100.0).collect();

        group.bench_with_input(
            BenchmarkId::new("euclidean", dimensions),
            &(a.clone(), b.clone()),
            |bench, (a, b)| {
                let metric = EuclideanDistance;
                bench.iter(|| black_box(metric.distance(a, b)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("cosine", dimensions),
            &(a.clone(), b.clone()),
            |bench, (a, b)| {
                let metric = CosineDistance;
                bench.iter(|| black_box(metric.distance(a, b)))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dot_product", dimensions),
            &(a.clone(), b.clone()),
            |bench, (a, b)| {
                let metric = DotProductSimilarity;
                bench.iter(|| black_box(metric.distance(a, b)))
            },
        );
    }

    group.finish();
}

fn bench_batch_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_operations");

    let query = vec![0.5; 768];
    let vectors: Vec<VectorData> = (0..1000)
        .map(|i| (0..768).map(|j| ((i + j) as f32) / 1000.0).collect())
        .collect();
    let vector_refs: Vec<&VectorData> = vectors.iter().collect();

    use veclite::distance::{DistanceMetric, EuclideanDistance};
    let metric = EuclideanDistance;

    group.bench_function("batch_distance_1000x768", |b| {
        b.iter(|| black_box(metric.batch_distance(&query, &vector_refs)))
    });

    // Compare with individual distance calculations
    group.bench_function("individual_distance_1000x768", |b| {
        b.iter(|| {
            let distances: Vec<f32> = vectors.iter().map(|v| metric.distance(&query, v)).collect();
            black_box(distances)
        })
    });

    group.finish();
}

fn bench_hnsw_vs_brute_force(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_vs_brute_force");

    let vector_counts = [100, 1000];
    let dimensions = 128;

    for &vector_count in &vector_counts {
        let vectors = create_test_vectors(vector_count, dimensions);
        let query = (0..dimensions)
            .map(|i| (i as f32) / 100.0)
            .collect::<Vec<f32>>();

        // Brute force configuration
        let brute_force_config = VecLiteConfig::default(); // Default is brute force

        // HNSW configuration
        let mut hnsw_config = VecLiteConfig::default();
        hnsw_config.index.index_type = IndexType::HNSW;
        hnsw_config.index.hnsw.ef_construction = 100;
        hnsw_config.index.hnsw.max_m = 16;
        hnsw_config.index.hnsw.ef_search = 50;

        group.throughput(Throughput::Elements(vector_count as u64));

        // Benchmark brute force search
        group.bench_with_input(
            BenchmarkId::new("brute_force_search", vector_count),
            &vectors,
            |b, vectors| {
                b.iter_with_setup(
                    || {
                        let db = VecLite::with_config(brute_force_config.clone()).unwrap();
                        for (id, vector, metadata) in vectors.iter() {
                            db.insert(id.clone(), vector.clone(), metadata.clone())
                                .unwrap();
                        }
                        db
                    },
                    |db| black_box(db.search(&query, 10).unwrap()),
                );
            },
        );

        // Benchmark HNSW search
        group.bench_with_input(
            BenchmarkId::new("hnsw_search", vector_count),
            &vectors,
            |b, vectors| {
                b.iter_with_setup(
                    || {
                        let db = VecLite::with_config(hnsw_config.clone()).unwrap();
                        for (id, vector, metadata) in vectors.iter() {
                            db.insert(id.clone(), vector.clone(), metadata.clone())
                                .unwrap();
                        }
                        db
                    },
                    |db| black_box(db.search(&query, 10).unwrap()),
                );
            },
        );
    }

    group.finish();
}

fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_construction");

    for vector_count in [100, 500, 1000].iter() {
        let vectors = create_test_vectors(*vector_count, 128);

        // Test different HNSW parameters
        let configs = vec![
            ("default", HNSWConfig::default()),
            (
                "high_quality",
                HNSWConfig {
                    m: 32,
                    max_m: 32,
                    max_m_l: 64,
                    ef_construction: 400,
                    ef_search: 100,
                    ml: 1.0 / (2.0_f64).ln(),
                },
            ),
            (
                "fast_build",
                HNSWConfig {
                    m: 8,
                    max_m: 8,
                    max_m_l: 16,
                    ef_construction: 50,
                    ef_search: 25,
                    ml: 1.0 / (2.0_f64).ln(),
                },
            ),
        ];

        for (config_name, hnsw_config) in configs {
            group.throughput(Throughput::Elements(*vector_count as u64));
            group.bench_with_input(
                BenchmarkId::new(config_name, vector_count),
                &vectors,
                |b, vectors| {
                    b.iter(|| {
                        let mut config = VecLiteConfig::default();
                        config.index.index_type = IndexType::HNSW;
                        config.index.hnsw = hnsw_config.clone();

                        let db = VecLite::with_config(config).unwrap();
                        for (id, vector, metadata) in vectors.iter() {
                            black_box(
                                db.insert(id.clone(), vector.clone(), metadata.clone())
                                    .unwrap(),
                            );
                        }
                    });
                },
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_vector_insertion,
    bench_vector_search,
    bench_distance_metrics,
    bench_batch_operations,
    bench_hnsw_vs_brute_force,
    bench_hnsw_construction
);
criterion_main!(benches);
