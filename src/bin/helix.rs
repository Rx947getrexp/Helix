//! Helix CLI - Command-line interface for Helix vector database
//!
//! This CLI provides a complete interface for managing Helix databases,
//! including vector operations, search, persistence, and performance benchmarking.

use clap::{Parser, Subcommand, ValueEnum};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use helix::{
    types::{Metadata, VectorData},
    Helix, HelixConfig,
};

#[derive(Parser)]
#[command(name = "helix")]
#[command(about = "A lightweight embeddable vector search database", long_about = None)]
#[command(version)]
struct Cli {
    /// Database file path
    #[arg(short, long, default_value = "database.vlt")]
    database: PathBuf,

    /// Verbosity level
    #[arg(short, long, action = clap::ArgAction::Count)]
    verbose: u8,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Database management commands
    #[command(subcommand)]
    Database(DatabaseCommands),

    /// Vector operations
    #[command(subcommand)]
    Vector(VectorCommands),

    /// Performance benchmarking
    Benchmark {
        /// Number of vectors to benchmark with
        #[arg(short, long, default_value_t = 10000)]
        count: usize,

        /// Vector dimensions
        #[arg(short, long, default_value_t = 128)]
        dimensions: usize,

        /// Benchmark type
        #[arg(short, long, value_enum, default_value_t = BenchmarkType::All)]
        benchmark_type: BenchmarkType,
    },
}

#[derive(Subcommand)]
enum DatabaseCommands {
    /// Create a new database
    Create {
        /// Maximum number of vectors
        #[arg(short, long, default_value_t = 1000000)]
        max_vectors: usize,

        /// Default k for searches
        #[arg(short, long, default_value_t = 10)]
        default_k: usize,

        /// Force overwrite existing database
        #[arg(short, long)]
        force: bool,
    },

    /// Show database information
    Info,

    /// Validate database integrity
    Validate,

    /// Compact database (remove deleted vectors)
    Compact,
}

#[derive(Subcommand)]
enum VectorCommands {
    /// Insert a vector
    Insert {
        /// Vector ID
        id: String,

        /// Vector data as JSON array or comma-separated values
        #[arg(short, long)]
        vector: String,

        /// Metadata as JSON object
        #[arg(short, long)]
        metadata: Option<String>,
    },

    /// Get a vector by ID
    Get {
        /// Vector ID
        id: String,

        /// Output format
        #[arg(short, long, value_enum, default_value_t = OutputFormat::Json)]
        format: OutputFormat,
    },

    /// Delete a vector by ID
    Delete {
        /// Vector ID
        id: String,
    },

    /// Search for similar vectors
    Search {
        /// Query vector as JSON array or comma-separated values
        query: String,

        /// Number of results to return
        #[arg(short, long, default_value_t = 10)]
        k: usize,

        /// Distance metric to use
        #[arg(short, long, default_value = "euclidean")]
        metric: String,

        /// Output format
        #[arg(short, long, value_enum, default_value_t = OutputFormat::Table)]
        format: OutputFormat,
    },

    /// Import vectors from a file
    Import {
        /// Input file path (JSON lines format)
        file: PathBuf,

        /// Batch size for imports
        #[arg(short, long, default_value_t = 1000)]
        batch_size: usize,
    },

    /// Export vectors to a file
    Export {
        /// Output file path
        file: PathBuf,

        /// Output format
        #[arg(short, long, value_enum, default_value_t = OutputFormat::JsonLines)]
        format: OutputFormat,
    },
}

#[derive(Clone, ValueEnum)]
enum OutputFormat {
    Json,
    Table,
    Csv,
    JsonLines,
}

#[derive(Clone, ValueEnum)]
enum BenchmarkType {
    Insert,
    Search,
    Mixed,
    All,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Set up logging based on verbosity
    setup_logging(cli.verbose);

    match cli.command {
        Commands::Database(db_cmd) => handle_database_command(db_cmd, &cli.database),
        Commands::Vector(vec_cmd) => handle_vector_command(vec_cmd, &cli.database),
        Commands::Benchmark {
            count,
            dimensions,
            benchmark_type,
        } => handle_benchmark_command(benchmark_type, count, dimensions, &cli.database),
    }
}

fn setup_logging(verbosity: u8) {
    match verbosity {
        0 => {} // No logging
        1 => println!("Verbose mode enabled"),
        2 => println!("Very verbose mode enabled"),
        _ => println!("Maximum verbosity enabled"),
    }
}

fn handle_database_command(
    cmd: DatabaseCommands,
    db_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        DatabaseCommands::Create {
            max_vectors,
            default_k,
            force,
        } => {
            if db_path.exists() && !force {
                eprintln!(
                    "Database already exists at {:?}. Use --force to overwrite.",
                    db_path
                );
                std::process::exit(1);
            }

            println!("Creating new database at {:?}", db_path);

            let mut config = HelixConfig::default();
            config.storage.max_vectors = max_vectors;
            config.query.default_k = default_k;

            let db = Helix::with_config(config)?;
            db.save(db_path)?;

            println!("✅ Database created successfully");
            println!("  Max vectors: {}", max_vectors);
            println!("  Default k: {}", default_k);
        }

        DatabaseCommands::Info => {
            if !db_path.exists() {
                eprintln!("Database does not exist at {:?}", db_path);
                std::process::exit(1);
            }

            let db = Helix::load(db_path)?;
            let stats = db.stats();
            let config = db.config();

            println!("📊 Database Information");
            println!("====================");
            println!("File: {:?}", db_path);
            println!("Vector count: {}", stats.vector_count);
            println!("Dimensions: {:?}", stats.dimensions.unwrap_or(0));
            println!("Total memory: {} bytes", stats.total_memory_bytes);
            println!("Average vector size: {} bytes", stats.average_vector_size);
            println!("Max vectors: {}", config.storage.max_vectors);
            println!("Default k: {}", config.query.default_k);
            println!("Empty: {}", db.is_empty());

            // Show available metrics
            let metrics = Helix::available_metrics();
            println!("Available metrics: {}", metrics.join(", "));
        }

        DatabaseCommands::Validate => {
            if !db_path.exists() {
                eprintln!("Database does not exist at {:?}", db_path);
                std::process::exit(1);
            }

            println!("Validating database...");
            let _db = Helix::load(db_path)?;
            println!("✅ Database validation passed");
        }

        DatabaseCommands::Compact => {
            if !db_path.exists() {
                eprintln!("Database does not exist at {:?}", db_path);
                std::process::exit(1);
            }

            println!("Compacting database...");
            let db = Helix::load(db_path)?;
            let old_size = fs::metadata(db_path)?.len();

            // Save the database (this effectively compacts it)
            db.save(db_path)?;

            let new_size = fs::metadata(db_path)?.len();
            let saved = old_size.saturating_sub(new_size);
            let percentage = if old_size > 0 {
                (saved as f64 / old_size as f64) * 100.0
            } else {
                0.0
            };

            println!("✅ Database compacted");
            println!("  Old size: {} bytes", old_size);
            println!("  New size: {} bytes", new_size);
            println!("  Saved: {} bytes ({:.1}%)", saved, percentage);
        }
    }
    Ok(())
}

fn handle_vector_command(
    cmd: VectorCommands,
    db_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    let db = if db_path.exists() {
        Helix::load(db_path)?
    } else {
        println!("Database does not exist, creating new one...");
        Helix::new()?
    };

    match cmd {
        VectorCommands::Insert {
            id,
            vector,
            metadata,
        } => {
            let vec_data = parse_vector(&vector)?;
            let meta_data = if let Some(meta) = metadata {
                parse_metadata(&meta)?
            } else {
                HashMap::new()
            };

            db.insert(id.clone(), vec_data, meta_data)?;
            db.save(db_path)?;

            println!("✅ Vector '{}' inserted successfully", id);
        }

        VectorCommands::Get { id, format } => match db.get(&id)? {
            Some(item) => match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "id": item.id,
                        "vector": item.vector,
                        "metadata": item.metadata,
                        "timestamp": item.timestamp
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                OutputFormat::Table => {
                    println!("📋 Vector Information");
                    println!("===================");
                    println!("ID: {}", item.id);
                    println!("Vector: {:?}", item.vector);
                    println!("Dimensions: {}", item.vector.len());
                    println!("Metadata: {:?}", item.metadata);
                    println!("Timestamp: {}", item.timestamp);
                }
                _ => {
                    println!("Format not supported for get command");
                }
            },
            None => {
                eprintln!("Vector '{}' not found", id);
                std::process::exit(1);
            }
        },

        VectorCommands::Delete { id } => {
            if db.delete(&id)? {
                db.save(db_path)?;
                println!("✅ Vector '{}' deleted successfully", id);
            } else {
                eprintln!("Vector '{}' not found", id);
                std::process::exit(1);
            }
        }

        VectorCommands::Search {
            query,
            k,
            metric,
            format,
        } => {
            let query_vec = parse_vector(&query)?;

            let start = Instant::now();
            let results = db.search_with_metric(&query_vec, k, &metric)?;
            let duration = start.elapsed();

            match format {
                OutputFormat::Json => {
                    let json = serde_json::json!({
                        "query": query_vec,
                        "k": k,
                        "metric": metric,
                        "duration_ms": duration.as_millis(),
                        "results": results.iter().map(|r| serde_json::json!({
                            "id": r.id,
                            "score": r.score,
                            "metadata": r.metadata
                        })).collect::<Vec<_>>()
                    });
                    println!("{}", serde_json::to_string_pretty(&json)?);
                }
                OutputFormat::Table => {
                    println!("🔍 Search Results");
                    println!("================");
                    println!("Query: {:?}", query_vec);
                    println!(
                        "Metric: {}, K: {}, Duration: {:.2}ms",
                        metric,
                        k,
                        duration.as_millis()
                    );
                    println!();

                    if results.is_empty() {
                        println!("No results found.");
                    } else {
                        println!("{:<20} {:<12} {}", "ID", "Score", "Metadata");
                        println!("{:-<50}", "");
                        for result in results {
                            let meta_str = if result.metadata.is_empty() {
                                "{}".to_string()
                            } else {
                                serde_json::to_string(&result.metadata)
                                    .unwrap_or_else(|_| "{}".to_string())
                            };
                            println!("{:<20} {:<12.6} {}", result.id, result.score, meta_str);
                        }
                    }
                }
                _ => {
                    println!("Format not supported for search command");
                }
            }
        }

        VectorCommands::Import { file, batch_size } => {
            let content = fs::read_to_string(&file)?;
            let mut count = 0;
            let mut batch = Vec::new();

            println!("Importing vectors from {:?}...", file);

            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }

                let json: Value = serde_json::from_str(line)?;
                let id = json["id"].as_str().unwrap().to_string();
                let vector: VectorData = json["vector"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_f64().unwrap() as f32)
                    .collect();

                let metadata: Metadata = if let Some(meta) = json.get("metadata") {
                    if let Some(obj) = meta.as_object() {
                        obj.iter()
                            .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
                            .collect()
                    } else {
                        HashMap::new()
                    }
                } else {
                    HashMap::new()
                };

                batch.push((id, vector, metadata));

                if batch.len() >= batch_size {
                    count += batch.len();
                    db.insert_batch(batch.clone())?;
                    batch.clear();

                    if count % 10000 == 0 {
                        println!("Imported {} vectors...", count);
                    }
                }
            }

            // Insert remaining vectors
            if !batch.is_empty() {
                count += batch.len();
                db.insert_batch(batch)?;
            }

            db.save(db_path)?;
            println!("✅ Imported {} vectors successfully", count);
        }

        VectorCommands::Export { file, format } => {
            println!("Exporting vectors to {:?}...", file);

            // This would need iteration support in Helix
            println!("Export functionality requires iteration support in Helix core library");
            println!("This is a placeholder for future implementation");
        }
    }

    Ok(())
}

fn handle_benchmark_command(
    benchmark_type: BenchmarkType,
    count: usize,
    dimensions: usize,
    db_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Helix Performance Benchmark");
    println!("================================");
    println!("Vectors: {}, Dimensions: {}", count, dimensions);
    println!();

    let db = Helix::new()?;

    match benchmark_type {
        BenchmarkType::Insert => benchmark_insert(&db, count, dimensions)?,
        BenchmarkType::Search => {
            populate_database(&db, count, dimensions)?;
            benchmark_search(&db, dimensions)?;
        }
        BenchmarkType::Mixed => {
            populate_database(&db, count / 2, dimensions)?;
            benchmark_mixed(&db, count / 2, dimensions)?;
        }
        BenchmarkType::All => {
            benchmark_insert(&db, count, dimensions)?;
            println!();
            benchmark_search(&db, dimensions)?;
            println!();
            benchmark_mixed(&db, count / 4, dimensions)?;
        }
    }

    // Save final database if specified
    if db_path != &PathBuf::from("database.vlt") {
        db.save(db_path)?;
        println!("Database saved to {:?}", db_path);
    }

    Ok(())
}

fn benchmark_insert(
    db: &Helix,
    count: usize,
    dimensions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📝 Insert Benchmark");
    println!("-----------------");

    let start = Instant::now();

    for i in 0..count {
        let vector: VectorData = (0..dimensions).map(|j| ((i + j) as f32) / 1000.0).collect();
        let metadata = [("index".to_string(), i.to_string())]
            .iter()
            .cloned()
            .collect();

        db.insert(format!("vec_{}", i), vector, metadata)?;

        if (i + 1) % 1000 == 0 {
            let elapsed = start.elapsed().as_millis() as f64 / 1000.0;
            let rate = (i + 1) as f64 / elapsed;
            println!(
                "  Inserted {}/{} vectors ({:.0} vectors/sec)",
                i + 1,
                count,
                rate
            );
        }
    }

    let duration = start.elapsed();
    let rate = count as f64 / duration.as_secs_f64();

    println!("✅ Insert benchmark completed");
    println!("  Total time: {:.2}s", duration.as_secs_f64());
    println!("  Average rate: {:.0} vectors/sec", rate);
    println!("  Final database size: {}", db.len());

    Ok(())
}

fn benchmark_search(db: &Helix, dimensions: usize) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 Search Benchmark");
    println!("-----------------");

    let query: VectorData = (0..dimensions).map(|i| (i as f32) / 1000.0).collect();
    let k_values = [1, 5, 10, 50, 100];
    let iterations = 100;

    for k in k_values {
        let start = Instant::now();

        for _ in 0..iterations {
            let _results = db.search(&query, k)?;
        }

        let duration = start.elapsed();
        let avg_time = duration.as_millis() as f64 / iterations as f64;
        let searches_per_sec = 1000.0 / avg_time;

        println!(
            "  k={:3}: {:.2}ms avg, {:.0} searches/sec",
            k, avg_time, searches_per_sec
        );
    }

    Ok(())
}

fn benchmark_mixed(
    db: &Helix,
    additional_count: usize,
    dimensions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("🔄 Mixed Workload Benchmark");
    println!("-------------------------");

    let operations = 1000;
    let insert_ratio = 0.3;
    let search_ratio = 0.6;
    let delete_ratio = 0.1;

    let inserts = (operations as f64 * insert_ratio) as usize;
    let searches = (operations as f64 * search_ratio) as usize;
    let deletes = operations - inserts - searches;

    println!(
        "Operations: {} inserts, {} searches, {} deletes",
        inserts, searches, deletes
    );

    let query: VectorData = (0..dimensions).map(|i| (i as f32) / 1000.0).collect();
    let start = Instant::now();

    // Mixed operations
    let mut op_count = 0;
    for i in 0..operations {
        if i < inserts {
            // Insert operation
            let vector: VectorData = (0..dimensions)
                .map(|j| ((additional_count + i + j) as f32) / 1000.0)
                .collect();
            let metadata = [("mixed".to_string(), "true".to_string())]
                .iter()
                .cloned()
                .collect();
            db.insert(format!("mixed_{}", i), vector, metadata)?;
        } else if i < inserts + searches {
            // Search operation
            let _results = db.search(&query, 10)?;
        } else {
            // Delete operation (try to delete existing vector)
            let _ = db.delete(&format!("vec_{}", op_count % 100));
            op_count += 1;
        }
    }

    let duration = start.elapsed();
    let ops_per_sec = operations as f64 / duration.as_secs_f64();

    println!("✅ Mixed workload completed");
    println!("  Total time: {:.2}s", duration.as_secs_f64());
    println!("  Average rate: {:.0} ops/sec", ops_per_sec);
    println!("  Final database size: {}", db.len());

    Ok(())
}

fn populate_database(
    db: &Helix,
    count: usize,
    dimensions: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Populating database with {} vectors...", count);

    for i in 0..count {
        let vector: VectorData = (0..dimensions).map(|j| ((i + j) as f32) / 1000.0).collect();
        let metadata = [("benchmark".to_string(), "search".to_string())]
            .iter()
            .cloned()
            .collect();

        db.insert(format!("vec_{}", i), vector, metadata)?;

        if (i + 1) % 5000 == 0 {
            println!("  Populated {}/{} vectors", i + 1, count);
        }
    }

    println!("Population complete");
    Ok(())
}

fn parse_vector(input: &str) -> Result<VectorData, Box<dyn std::error::Error>> {
    if input.starts_with('[') && input.ends_with(']') {
        // JSON array format
        let json: Value = serde_json::from_str(input)?;
        Ok(json
            .as_array()
            .ok_or("Invalid JSON array")?
            .iter()
            .map(|v| v.as_f64().unwrap_or(0.0) as f32)
            .collect())
    } else {
        // Comma-separated values
        Ok(input
            .split(',')
            .map(|s| s.trim().parse::<f32>().unwrap_or(0.0))
            .collect())
    }
}

fn parse_metadata(input: &str) -> Result<Metadata, Box<dyn std::error::Error>> {
    let json: Value = serde_json::from_str(input)?;
    let obj = json.as_object().ok_or("Metadata must be a JSON object")?;

    Ok(obj
        .iter()
        .map(|(k, v)| (k.clone(), v.as_str().unwrap_or("").to_string()))
        .collect())
}
