// Package main demonstrates performance characteristics of VecLite Go bindings.
package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"veclite"
)

func main() {
	fmt.Println("VecLite Performance Demonstration")
	fmt.Println("=================================")

	// Create a new VecLite instance with custom config
	fmt.Println("\n1. Creating VecLite database with custom config...")
	config := &veclite.Config{
		MaxVectors: 100000,
		DefaultK:   10,
	}

	db, err := veclite.NewWithConfig(config)
	if err != nil {
		log.Fatalf("Failed to create VecLite instance: %v", err)
	}
	defer db.Close()

	// Test insertion performance
	fmt.Println("\n2. Testing insertion performance...")
	vectorCount := 10000
	dimensions := 128

	fmt.Printf("Inserting %d vectors with %d dimensions...\n", vectorCount, dimensions)

	start := time.Now()

	for i := 0; i < vectorCount; i++ {
		// Generate random vector
		vector := make(veclite.Vector, dimensions)
		for j := 0; j < dimensions; j++ {
			vector[j] = rand.Float32()*2 - 1 // Random values between -1 and 1
		}

		metadata := veclite.Metadata{
			"index":    fmt.Sprintf("%d", i),
			"category": fmt.Sprintf("cat_%d", i%10),
			"source":   "synthetic",
		}

		id := fmt.Sprintf("vec_%d", i)
		err := db.Insert(id, vector, metadata)
		if err != nil {
			log.Fatalf("Failed to insert vector %d: %v", i, err)
		}

		// Print progress every 1000 vectors
		if (i+1)%1000 == 0 {
			fmt.Printf("  Inserted %d/%d vectors\n", i+1, vectorCount)
		}
	}

	insertionDuration := time.Since(start)
	insertionsPerSecond := float64(vectorCount) / insertionDuration.Seconds()

	fmt.Printf("✅ Insertion completed!\n")
	fmt.Printf("  Total time: %v\n", insertionDuration)
	fmt.Printf("  Insertions per second: %.0f\n", insertionsPerSecond)
	fmt.Printf("  Database length: %d\n", db.Len())

	// Test search performance
	fmt.Println("\n3. Testing search performance...")

	// Generate random query vector
	queryVector := make(veclite.Vector, dimensions)
	for j := 0; j < dimensions; j++ {
		queryVector[j] = rand.Float32()*2 - 1
	}

	searchCount := 100
	kValues := []uint{1, 5, 10, 50, 100}

	for _, k := range kValues {
		fmt.Printf("\nTesting search with k=%d (%d searches)...\n", k, searchCount)

		start = time.Now()

		for i := 0; i < searchCount; i++ {
			results, err := db.Search(queryVector, k)
			if err != nil {
				log.Fatalf("Search %d failed: %v", i, err)
			}

			if len(results) != int(k) && len(results) != int(db.Len()) {
				log.Fatalf("Expected %d results, got %d", k, len(results))
			}
		}

		searchDuration := time.Since(start)
		avgSearchTime := searchDuration / time.Duration(searchCount)
		searchesPerSecond := float64(searchCount) / searchDuration.Seconds()

		fmt.Printf("  Average search time: %v\n", avgSearchTime)
		fmt.Printf("  Searches per second: %.0f\n", searchesPerSecond)
	}

	// Test memory usage and database statistics
	fmt.Println("\n4. Database statistics...")
	fmt.Printf("  Total vectors: %d\n", db.Len())
	fmt.Printf("  Empty: %v\n", db.IsEmpty())

	// Test persistence performance
	fmt.Println("\n5. Testing persistence performance...")

	dbPath := "performance_test.vlt"
	fmt.Printf("Saving database to %s...\n", dbPath)

	start = time.Now()
	err = db.Save(dbPath)
	if err != nil {
		log.Fatalf("Failed to save database: %v", err)
	}
	saveDuration := time.Since(start)

	fmt.Printf("✅ Save completed in %v\n", saveDuration)

	// Test loading performance
	fmt.Printf("Loading database from %s...\n", dbPath)

	start = time.Now()
	loadedDB, err := veclite.Load(dbPath)
	if err != nil {
		log.Fatalf("Failed to load database: %v", err)
	}
	loadDuration := time.Since(start)

	fmt.Printf("✅ Load completed in %v\n", loadDuration)
	fmt.Printf("  Loaded vectors: %d\n", loadedDB.Len())

	// Verify loaded database with a search
	results, err := loadedDB.Search(queryVector, 5)
	if err != nil {
		log.Fatalf("Search on loaded database failed: %v", err)
	}
	fmt.Printf("  Verification search returned %d results\n", len(results))

	loadedDB.Close()

	// Test batch operations simulation
	fmt.Println("\n6. Simulating real-world usage patterns...")

	// Mixed workload: 70% searches, 20% insertions, 10% deletions
	workloadSize := 1000
	searchOps := int(0.7 * float64(workloadSize))
	insertOps := int(0.2 * float64(workloadSize))
	deleteOps := workloadSize - searchOps - insertOps

	fmt.Printf("Mixed workload: %d searches, %d inserts, %d deletes\n",
		searchOps, insertOps, deleteOps)

	start = time.Now()

	operations := make([]string, workloadSize)
	for i := 0; i < searchOps; i++ {
		operations[i] = "search"
	}
	for i := searchOps; i < searchOps+insertOps; i++ {
		operations[i] = "insert"
	}
	for i := searchOps+insertOps; i < workloadSize; i++ {
		operations[i] = "delete"
	}

	// Shuffle operations
	rand.Shuffle(len(operations), func(i, j int) {
		operations[i], operations[j] = operations[j], operations[i]
	})

	searchCount = 0
	insertCount := 0
	deleteCount := 0

	for i, op := range operations {
		switch op {
		case "search":
			_, err := db.Search(queryVector, 10)
			if err != nil {
				log.Printf("Search operation %d failed: %v", i, err)
			} else {
				searchCount++
			}

		case "insert":
			vector := make(veclite.Vector, dimensions)
			for j := 0; j < dimensions; j++ {
				vector[j] = rand.Float32()*2 - 1
			}
			metadata := veclite.Metadata{
				"workload": "mixed",
				"op_index": fmt.Sprintf("%d", i),
			}

			id := fmt.Sprintf("mixed_vec_%d", vectorCount+insertCount)
			err := db.Insert(id, vector, metadata)
			if err != nil {
				log.Printf("Insert operation %d failed: %v", i, err)
			} else {
				insertCount++
			}

		case "delete":
			// Try to delete an existing vector
			deleteIndex := rand.Intn(vectorCount)
			id := fmt.Sprintf("vec_%d", deleteIndex)
			err := db.Delete(id)
			if err == nil {
				deleteCount++
			}
			// Ignore errors since vector might already be deleted
		}

		if (i+1)%100 == 0 {
			fmt.Printf("  Completed %d/%d operations\n", i+1, workloadSize)
		}
	}

	mixedDuration := time.Since(start)
	opsPerSecond := float64(workloadSize) / mixedDuration.Seconds()

	fmt.Printf("✅ Mixed workload completed!\n")
	fmt.Printf("  Total time: %v\n", mixedDuration)
	fmt.Printf("  Operations per second: %.0f\n", opsPerSecond)
	fmt.Printf("  Successful searches: %d\n", searchCount)
	fmt.Printf("  Successful inserts: %d\n", insertCount)
	fmt.Printf("  Successful deletes: %d\n", deleteCount)
	fmt.Printf("  Final database size: %d\n", db.Len())

	fmt.Println("\n🎯 Performance test completed successfully!")
	fmt.Println("\nKey metrics:")
	fmt.Printf("  - Insertion rate: %.0f vectors/sec\n", insertionsPerSecond)
	fmt.Printf("  - Mixed workload: %.0f ops/sec\n", opsPerSecond)
	fmt.Printf("  - Save time: %v\n", saveDuration)
	fmt.Printf("  - Load time: %v\n", loadDuration)
}