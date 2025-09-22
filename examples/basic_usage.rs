//! Basic Helix Usage Example
//!
//! This example demonstrates the fundamental operations of Helix:
//! - Creating a database instance
//! - Inserting vectors with metadata
//! - Performing basic searches
//! - Working with results
//!
//! Run with: cargo run --example basic_usage

use std::collections::HashMap;
use helix::{Helix, HelixConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Helix Basic Usage Example");
    println!("==============================\n");

    // Step 1: Create a Helix instance
    println!("📊 Creating Helix database...");
    let db = Helix::new()?;
    println!("✅ Database created successfully\n");

    // Step 2: Prepare some sample vectors and metadata
    println!("📝 Preparing sample data...");
    let samples = vec![
        (
            "doc1",
            vec![1.0, 0.5, 0.8],
            "This is about machine learning and AI",
            "technology"
        ),
        (
            "doc2",
            vec![0.2, 1.0, 0.3],
            "Cooking recipes and food preparation",
            "food"
        ),
        (
            "doc3",
            vec![0.9, 0.4, 1.0],
            "Latest developments in quantum computing",
            "technology"
        ),
        (
            "doc4",
            vec![0.1, 0.8, 0.2],
            "Healthy eating and nutrition tips",
            "food"
        ),
        (
            "doc5",
            vec![0.7, 0.1, 0.9],
            "Artificial intelligence and neural networks",
            "technology"
        ),
    ];

    // Step 3: Insert vectors with metadata
    println!("📥 Inserting vectors into database...");
    for (id, vector, description, category) in &samples {
        let metadata = HashMap::from([
            ("description".to_string(), description.to_string()),
            ("category".to_string(), category.to_string()),
            ("source".to_string(), "example_dataset".to_string()),
        ]);

        db.insert(id.to_string(), vector.clone(), metadata)?;
        println!("   ✓ Inserted vector: {}", id);
    }

    println!("✅ Inserted {} vectors\n", samples.len());

    // Step 4: Check database statistics
    println!("📈 Database Statistics:");
    println!("   📊 Total vectors: {}", db.len());
    println!("   📏 Dimensions: {:?}", db.stats().dimensions);
    println!("   💾 Memory usage: {} bytes", db.stats().memory.current_allocated);
    println!();

    // Step 5: Perform basic search
    println!("🔍 Performing vector search...");
    let query_vector = vec![0.8, 0.3, 0.9]; // Similar to technology vectors
    println!("   🎯 Query vector: {:?}", query_vector);

    let search_results = db.search(&query_vector, 3)?;

    println!("   📋 Found {} results:", search_results.len());
    for (i, result) in search_results.iter().enumerate() {
        println!("   {}. ID: {} (Score: {:.4})", i + 1, result.id, result.score);
        if let Some(description) = result.metadata.get("description") {
            println!("      Description: {}", description);
        }
        if let Some(category) = result.metadata.get("category") {
            println!("      Category: {}", category);
        }
        println!();
    }

    // Step 6: Try different distance metrics
    println!("🧮 Comparing different distance metrics...");
    let metrics = vec!["euclidean", "cosine", "dot_product"];

    for metric in metrics {
        println!("   📐 Using {} distance:", metric);
        match db.search_with_metric(&query_vector, 2, metric) {
            Ok(results) => {
                for result in results {
                    println!("      • {} (Score: {:.4})", result.id, result.score);
                }
            }
            Err(e) => {
                println!("      ❌ Error with {}: {}", metric, e);
            }
        }
        println!();
    }

    // Step 7: Search with metadata filtering
    println!("🎛️  Searching with metadata filter...");
    let filtered_results = db.search_with_filter(
        &query_vector,
        5,
        |metadata| metadata.get("category") == Some(&"technology".to_string())
    )?;

    println!("   🔍 Technology documents only:");
    for result in filtered_results {
        println!("      • {} (Score: {:.4})", result.id, result.score);
        if let Some(description) = result.metadata.get("description") {
            println!("        {}", description);
        }
    }
    println!();

    // Step 8: Retrieve specific vectors
    println!("📖 Retrieving specific vectors...");
    if let Some(vector_item) = db.get(&"doc1".to_string())? {
        println!("   📄 Retrieved doc1:");
        println!("      Vector: {:?}", vector_item.vector);
        println!("      Metadata: {:?}", vector_item.metadata);
        println!("      Timestamp: {}", vector_item.timestamp);
    }
    println!();

    // Step 9: Batch operations example
    println!("📦 Demonstrating batch operations...");
    let batch_vectors = vec![
        (
            "batch1".to_string(),
            vec![0.3, 0.7, 0.5],
            HashMap::from([("type".to_string(), "batch".to_string())])
        ),
        (
            "batch2".to_string(),
            vec![0.6, 0.2, 0.8],
            HashMap::from([("type".to_string(), "batch".to_string())])
        ),
    ];

    db.insert_batch(batch_vectors)?;
    println!("   ✅ Inserted {} vectors in batch", 2);
    println!("   📊 Total vectors now: {}", db.len());
    println!();

    // Step 10: Database persistence
    println!("💾 Demonstrating persistence...");
    let save_path = "example_database.vlt";

    // Save the database
    db.save(save_path)?;
    println!("   ✅ Database saved to: {}", save_path);

    // Load the database (create new instance)
    let loaded_db = Helix::open(save_path)?;
    println!("   ✅ Database loaded successfully");
    println!("   📊 Loaded vectors: {}", loaded_db.len());

    // Verify data integrity
    let verification_results = loaded_db.search(&query_vector, 2)?;
    println!("   🔍 Verification search returned {} results", verification_results.len());
    println!();

    // Step 11: Configuration example
    println!("⚙️  Custom configuration example...");
    let mut config = HelixConfig::default();
    config.storage.max_vectors = 10_000;
    config.query.default_k = 5;
    config.index.index_type = helix::IndexType::HNSW;

    let custom_db = Helix::with_config(config)?;
    println!("   ✅ Created database with custom configuration");
    println!("   📋 Max vectors: {}", custom_db.config().storage.max_vectors);
    println!("   🎯 Default K: {}", custom_db.config().query.default_k);
    println!();

    // Clean up
    println!("🧹 Cleaning up...");
    if std::path::Path::new(save_path).exists() {
        std::fs::remove_file(save_path)?;
        println!("   ✅ Cleaned up temporary file");
    }

    println!("🎉 Basic usage example completed successfully!");
    println!("\n💡 Next steps:");
    println!("   • Try the advanced_search example");
    println!("   • Check out performance_benchmark example");
    println!("   • Read the documentation at docs/guides/TUTORIAL.md");

    Ok(())
}