//! Advanced Helix Search Example
//!
//! This example demonstrates advanced search capabilities of Helix:
//! - Complex metadata filtering
//! - Multiple distance metrics comparison
//! - Batch search operations
//! - Custom search configurations
//! - Result ranking and analysis
//!
//! Run with: cargo run --example advanced_search

use helix::{Helix, IndexType, VecLiteConfig};
use std::collections::HashMap;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Helix Advanced Search Example");
    println!("==================================\n");

    // Step 1: Create database with optimized configuration
    println!("⚙️  Setting up optimized Helix configuration...");
    let mut config = VecLiteConfig::default();
    config.index.index_type = IndexType::HNSW;
    config.index.hnsw.ef_construction = 200;
    config.index.hnsw.ef_search = 100;
    config.index.hnsw.max_m = 16;
    config.query.default_k = 10;

    let db = Helix::with_config(config)?;
    println!("✅ Database configured with HNSW index\n");

    // Step 2: Create a comprehensive dataset
    println!("📚 Creating comprehensive dataset...");
    let documents = create_document_dataset();

    println!("📥 Inserting {} documents...", documents.len());
    db.insert_batch(documents)?;
    println!("✅ Dataset inserted successfully\n");

    // Step 3: Advanced metadata filtering
    println!("🎛️  Advanced Metadata Filtering");
    println!("-------------------------------");

    demonstrate_complex_filtering(&db)?;

    // Step 4: Distance metric comparison
    println!("\n📐 Distance Metric Comparison");
    println!("----------------------------");

    demonstrate_metric_comparison(&db)?;

    // Step 5: Batch search operations
    println!("\n📦 Batch Search Operations");
    println!("-------------------------");

    demonstrate_batch_search(&db)?;

    // Step 6: Search result analysis
    println!("\n📊 Search Result Analysis");
    println!("------------------------");

    demonstrate_result_analysis(&db)?;

    // Step 7: Performance-tuned searches
    println!("\n⚡ Performance-Tuned Searches");
    println!("-----------------------------");

    demonstrate_performance_tuning(&db)?;

    // Step 8: Similarity threshold filtering
    println!("\n🎯 Similarity Threshold Filtering");
    println!("--------------------------------");

    demonstrate_threshold_filtering(&db)?;

    println!("\n🎉 Advanced search example completed!");
    println!("\n💡 Key takeaways:");
    println!("   • Use appropriate distance metrics for your data");
    println!("   • Leverage metadata filtering for precise results");
    println!("   • Batch operations improve performance");
    println!("   • Tune HNSW parameters for your use case");
    println!("   • Apply similarity thresholds to filter results");

    Ok(())
}

fn create_document_dataset() -> Vec<(String, Vec<f32>, HashMap<String, String>)> {
    vec![
        // Technology documents
        (
            "tech_001".to_string(),
            vec![0.9, 0.1, 0.8, 0.2, 0.7],
            create_metadata(
                "Machine Learning Fundamentals",
                "technology",
                "ai",
                2023,
                4.5,
                "en",
            ),
        ),
        (
            "tech_002".to_string(),
            vec![0.8, 0.2, 0.9, 0.1, 0.6],
            create_metadata(
                "Deep Learning with PyTorch",
                "technology",
                "ai",
                2023,
                4.7,
                "en",
            ),
        ),
        (
            "tech_003".to_string(),
            vec![0.7, 0.3, 0.7, 0.4, 0.8],
            create_metadata(
                "Computer Vision Applications",
                "technology",
                "ai",
                2022,
                4.3,
                "en",
            ),
        ),
        (
            "tech_004".to_string(),
            vec![0.6, 0.1, 0.5, 0.8, 0.4],
            create_metadata(
                "Web Development with React",
                "technology",
                "web",
                2023,
                4.2,
                "en",
            ),
        ),
        (
            "tech_005".to_string(),
            vec![0.5, 0.2, 0.6, 0.7, 0.5],
            create_metadata(
                "Database Design Principles",
                "technology",
                "database",
                2022,
                4.0,
                "en",
            ),
        ),
        // Science documents
        (
            "sci_001".to_string(),
            vec![0.2, 0.9, 0.3, 0.8, 0.1],
            create_metadata(
                "Quantum Physics Explained",
                "science",
                "physics",
                2023,
                4.8,
                "en",
            ),
        ),
        (
            "sci_002".to_string(),
            vec![0.3, 0.8, 0.2, 0.9, 0.2],
            create_metadata(
                "Climate Change Impact",
                "science",
                "environment",
                2023,
                4.6,
                "en",
            ),
        ),
        (
            "sci_003".to_string(),
            vec![0.1, 0.7, 0.4, 0.6, 0.3],
            create_metadata(
                "Genetic Engineering Advances",
                "science",
                "biology",
                2022,
                4.4,
                "en",
            ),
        ),
        (
            "sci_004".to_string(),
            vec![0.4, 0.6, 0.1, 0.7, 0.4],
            create_metadata(
                "Space Exploration Timeline",
                "science",
                "astronomy",
                2023,
                4.5,
                "en",
            ),
        ),
        // Health documents
        (
            "health_001".to_string(),
            vec![0.1, 0.3, 0.9, 0.4, 0.8],
            create_metadata(
                "Nutrition and Wellness Guide",
                "health",
                "nutrition",
                2023,
                4.1,
                "en",
            ),
        ),
        (
            "health_002".to_string(),
            vec![0.2, 0.4, 0.8, 0.5, 0.9],
            create_metadata(
                "Exercise for Mental Health",
                "health",
                "fitness",
                2022,
                4.3,
                "en",
            ),
        ),
        (
            "health_003".to_string(),
            vec![0.3, 0.2, 0.7, 0.3, 0.6],
            create_metadata("Sleep and Recovery", "health", "wellness", 2023, 4.0, "en"),
        ),
        // Arts documents
        (
            "art_001".to_string(),
            vec![0.8, 0.7, 0.2, 0.1, 0.3],
            create_metadata(
                "Renaissance Art History",
                "arts",
                "painting",
                2022,
                4.7,
                "en",
            ),
        ),
        (
            "art_002".to_string(),
            vec![0.7, 0.6, 0.3, 0.2, 0.4],
            create_metadata(
                "Modern Photography Techniques",
                "arts",
                "photography",
                2023,
                4.2,
                "en",
            ),
        ),
        (
            "art_003".to_string(),
            vec![0.6, 0.8, 0.1, 0.3, 0.2],
            create_metadata("Music Theory Basics", "arts", "music", 2022, 4.4, "en"),
        ),
        // Business documents
        (
            "biz_001".to_string(),
            vec![0.4, 0.5, 0.6, 0.9, 0.7],
            create_metadata(
                "Startup Business Strategy",
                "business",
                "strategy",
                2023,
                4.3,
                "en",
            ),
        ),
        (
            "biz_002".to_string(),
            vec![0.5, 0.4, 0.7, 0.8, 0.8],
            create_metadata(
                "Digital Marketing Trends",
                "business",
                "marketing",
                2023,
                4.1,
                "en",
            ),
        ),
        (
            "biz_003".to_string(),
            vec![0.3, 0.6, 0.5, 0.7, 0.6],
            create_metadata(
                "Financial Planning Guide",
                "business",
                "finance",
                2022,
                4.5,
                "en",
            ),
        ),
        // International content
        (
            "intl_001".to_string(),
            vec![0.9, 0.2, 0.7, 0.3, 0.8],
            create_metadata("机器学习入门", "technology", "ai", 2023, 4.6, "zh"),
        ),
        (
            "intl_002".to_string(),
            vec![0.2, 0.8, 0.3, 0.7, 0.1],
            create_metadata("Física Cuántica", "science", "physics", 2023, 4.4, "es"),
        ),
    ]
}

fn create_metadata(
    title: &str,
    category: &str,
    subcategory: &str,
    year: i32,
    rating: f32,
    language: &str,
) -> HashMap<String, String> {
    HashMap::from([
        ("title".to_string(), title.to_string()),
        ("category".to_string(), category.to_string()),
        ("subcategory".to_string(), subcategory.to_string()),
        ("year".to_string(), year.to_string()),
        ("rating".to_string(), rating.to_string()),
        ("language".to_string(), language.to_string()),
        (
            "indexed_at".to_string(),
            chrono::Utc::now().timestamp().to_string(),
        ),
    ])
}

fn demonstrate_complex_filtering(db: &Helix) -> Result<(), Box<dyn std::error::Error>> {
    let query = vec![0.8, 0.2, 0.6, 0.3, 0.7]; // Tech-oriented query

    // Filter 1: High-rated technology documents from 2023
    println!("🔍 Filter 1: High-rated 2023 technology documents (rating >= 4.5)");
    let results = db.search_with_filter(&query, 5, |metadata| {
        metadata.get("category") == Some(&"technology".to_string())
            && metadata.get("year") == Some(&"2023".to_string())
            && metadata
                .get("rating")
                .and_then(|r| r.parse::<f32>().ok())
                .unwrap_or(0.0)
                >= 4.5
    })?;

    print_search_results(&results, "High-rated Tech 2023");

    // Filter 2: Multilingual science content
    println!("\n🔍 Filter 2: Science documents in any language");
    let results = db.search_with_filter(&query, 10, |metadata| {
        metadata.get("category") == Some(&"science".to_string())
    })?;

    print_search_results(&results, "Science (All Languages)");

    // Filter 3: AI-related content across categories
    println!("\n🔍 Filter 3: AI-related content (any category)");
    let results = db.search_with_filter(&query, 10, |metadata| {
        metadata.get("subcategory") == Some(&"ai".to_string())
            || metadata
                .get("title")
                .map(|t| {
                    t.to_lowercase().contains("ai")
                        || t.to_lowercase().contains("machine learning")
                        || t.to_lowercase().contains("deep learning")
                })
                .unwrap_or(false)
    })?;

    print_search_results(&results, "AI-Related Content");

    Ok(())
}

fn demonstrate_metric_comparison(db: &Helix) -> Result<(), Box<dyn std::error::Error>> {
    let query = vec![0.7, 0.3, 0.8, 0.2, 0.6];
    let metrics = vec!["euclidean", "cosine", "dot_product", "manhattan"];

    println!("📊 Comparing search results across different distance metrics:");
    println!("Query vector: {:?}\n", query);

    for metric in metrics {
        println!("📐 {} Distance:", metric.to_uppercase());

        let start_time = Instant::now();
        match db.search_with_metric(&query, 3, metric) {
            Ok(results) => {
                let duration = start_time.elapsed();
                println!("   ⏱️  Search time: {:?}", duration);

                for (i, result) in results.iter().enumerate() {
                    if let Some(title) = result.metadata.get("title") {
                        println!("   {}. {} (Score: {:.4})", i + 1, title, result.score);
                    } else {
                        println!("   {}. {} (Score: {:.4})", i + 1, result.id, result.score);
                    }
                }
            }
            Err(e) => {
                println!("   ❌ Error: {}", e);
            }
        }
        println!();
    }

    Ok(())
}

fn demonstrate_batch_search(db: &Helix) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Performing batch search operations...");

    // Create multiple query vectors
    let queries = vec![
        ("Tech Query", vec![0.9, 0.1, 0.7, 0.2, 0.8]), // Technology-focused
        ("Science Query", vec![0.2, 0.9, 0.3, 0.8, 0.1]), // Science-focused
        ("Health Query", vec![0.1, 0.3, 0.9, 0.4, 0.7]), // Health-focused
        ("Arts Query", vec![0.8, 0.7, 0.2, 0.1, 0.3]), // Arts-focused
    ];

    let start_time = Instant::now();

    // Extract just the vectors for batch search
    let query_vectors: Vec<Vec<f32>> = queries.iter().map(|(_, v)| v.clone()).collect();
    let batch_results = db.batch_search(&query_vectors, 3)?;

    let batch_duration = start_time.elapsed();
    println!("⏱️  Batch search completed in: {:?}", batch_duration);
    println!("📊 Processed {} queries simultaneously\n", queries.len());

    // Display results for each query
    for (i, (query_name, _)) in queries.iter().enumerate() {
        println!("🎯 Results for {}:", query_name);
        if let Some(results) = batch_results.get(i) {
            for (j, result) in results.iter().enumerate() {
                if let Some(title) = result.metadata.get("title") {
                    println!("   {}. {} (Score: {:.4})", j + 1, title, result.score);
                } else {
                    println!("   {}. {} (Score: {:.4})", j + 1, result.id, result.score);
                }
            }
        }
        println!();
    }

    // Compare with individual searches
    println!("⚖️  Comparing batch vs individual search performance:");
    let individual_start = Instant::now();
    for (_, query_vector) in &queries {
        let _ = db.search(query_vector, 3)?;
    }
    let individual_duration = individual_start.elapsed();

    println!("   📦 Batch search: {:?}", batch_duration);
    println!("   🔍 Individual searches: {:?}", individual_duration);
    println!(
        "   🚀 Speedup: {:.2}x",
        individual_duration.as_secs_f64() / batch_duration.as_secs_f64()
    );

    Ok(())
}

fn demonstrate_result_analysis(db: &Helix) -> Result<(), Box<dyn std::error::Error>> {
    let query = vec![0.8, 0.2, 0.6, 0.4, 0.7];

    println!("📈 Analyzing search results...");
    let results = db.search(&query, 10)?;

    // Score distribution analysis
    let scores: Vec<f32> = results.iter().map(|r| r.score).collect();
    let min_score = scores.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_score = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let avg_score = scores.iter().sum::<f32>() / scores.len() as f32;

    println!("📊 Score Statistics:");
    println!("   📉 Min score: {:.4}", min_score);
    println!("   📈 Max score: {:.4}", max_score);
    println!("   📊 Average score: {:.4}", avg_score);
    println!("   📏 Score range: {:.4}", max_score - min_score);

    // Category distribution
    let mut category_counts: HashMap<String, usize> = HashMap::new();
    for result in &results {
        if let Some(category) = result.metadata.get("category") {
            *category_counts.entry(category.clone()).or_insert(0) += 1;
        }
    }

    println!("\n📋 Category Distribution:");
    for (category, count) in category_counts {
        println!("   📂 {}: {} documents", category, count);
    }

    // Rating analysis
    let ratings: Vec<f32> = results
        .iter()
        .filter_map(|r| r.metadata.get("rating"))
        .filter_map(|r| r.parse::<f32>().ok())
        .collect();

    if !ratings.is_empty() {
        let avg_rating = ratings.iter().sum::<f32>() / ratings.len() as f32;
        println!("\n⭐ Average rating of results: {:.2}", avg_rating);
    }

    // Language distribution
    let mut lang_counts: HashMap<String, usize> = HashMap::new();
    for result in &results {
        if let Some(lang) = result.metadata.get("language") {
            *lang_counts.entry(lang.clone()).or_insert(0) += 1;
        }
    }

    println!("\n🌍 Language Distribution:");
    for (lang, count) in lang_counts {
        println!("   🗣️  {}: {} documents", lang, count);
    }

    Ok(())
}

fn demonstrate_performance_tuning(db: &Helix) -> Result<(), Box<dyn std::error::Error>> {
    let query = vec![0.7, 0.3, 0.8, 0.2, 0.6];

    println!("⚙️  Testing different search configurations...");

    // Test different K values
    let k_values = vec![1, 3, 5, 10, 15];

    println!("\n📊 Performance vs K value:");
    for k in k_values {
        let start_time = Instant::now();
        let results = db.search(&query, k)?;
        let duration = start_time.elapsed();

        println!(
            "   K={:2} | Time: {:>8?} | Results: {:2}",
            k,
            duration,
            results.len()
        );
    }

    // Memory usage analysis
    let stats = db.stats();
    println!("\n💾 Memory Usage:");
    println!(
        "   📊 Current allocated: {} bytes",
        stats.memory.current_allocated
    );
    println!(
        "   📈 Peak allocated: {} bytes",
        stats.memory.peak_allocated
    );
    println!(
        "   🔄 Total allocations: {}",
        stats.memory.total_allocations
    );

    Ok(())
}

fn demonstrate_threshold_filtering(db: &Helix) -> Result<(), Box<dyn std::error::Error>> {
    let query = vec![0.8, 0.2, 0.6, 0.3, 0.7];

    println!("🎯 Demonstrating similarity threshold filtering...");

    // Get all results first
    let all_results = db.search(&query, 20)?;

    // Apply different similarity thresholds
    let thresholds = vec![0.1, 0.3, 0.5, 0.7, 0.9];

    for threshold in thresholds {
        // Filter results based on threshold (assuming lower scores are better for distance metrics)
        let filtered_results: Vec<_> = all_results
            .iter()
            .filter(|r| r.score <= threshold)
            .collect();

        println!(
            "\n📏 Threshold <= {:.1}: {} results",
            threshold,
            filtered_results.len()
        );

        if !filtered_results.is_empty() {
            for (i, result) in filtered_results.iter().take(3).enumerate() {
                if let Some(title) = result.metadata.get("title") {
                    println!("   {}. {} (Score: {:.4})", i + 1, title, result.score);
                }
            }

            if filtered_results.len() > 3 {
                println!("   ... and {} more", filtered_results.len() - 3);
            }
        }
    }

    Ok(())
}

fn print_search_results(results: &[helix::SearchResult], label: &str) {
    println!("   📋 {} ({} results):", label, results.len());
    for (i, result) in results.iter().take(3).enumerate() {
        if let Some(title) = result.metadata.get("title") {
            println!("      {}. {} (Score: {:.4})", i + 1, title, result.score);
            if let Some(category) = result.metadata.get("category") {
                println!("         Category: {}", category);
            }
        }
    }
    if results.len() > 3 {
        println!("      ... and {} more results", results.len() - 3);
    }
}
