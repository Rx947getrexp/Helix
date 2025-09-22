// Package main demonstrates basic usage of the VecLite Go bindings.
package main

import (
	"fmt"
	"log"

	"veclite"
)

func main() {
	fmt.Println("VecLite Go Bindings Example")
	fmt.Println("===========================")

	// Create a new VecLite instance
	fmt.Println("\n1. Creating VecLite database...")
	db, err := veclite.New()
	if err != nil {
		log.Fatalf("Failed to create VecLite instance: %v", err)
	}
	defer db.Close()

	fmt.Printf("Database created successfully. Empty: %v, Length: %d\n", db.IsEmpty(), db.Len())

	// Insert some sample vectors with metadata
	fmt.Println("\n2. Inserting sample vectors...")
	vectors := []struct {
		id       string
		vector   veclite.Vector
		metadata veclite.Metadata
	}{
		{
			id:     "doc1",
			vector: veclite.Vector{1.0, 0.0, 0.0},
			metadata: veclite.Metadata{
				"type":   "document",
				"title":  "First Document",
				"author": "Alice",
			},
		},
		{
			id:     "doc2",
			vector: veclite.Vector{0.0, 1.0, 0.0},
			metadata: veclite.Metadata{
				"type":   "document",
				"title":  "Second Document",
				"author": "Bob",
			},
		},
		{
			id:     "doc3",
			vector: veclite.Vector{0.0, 0.0, 1.0},
			metadata: veclite.Metadata{
				"type":   "document",
				"title":  "Third Document",
				"author": "Charlie",
			},
		},
		{
			id:     "img1",
			vector: veclite.Vector{0.7, 0.7, 0.0},
			metadata: veclite.Metadata{
				"type":   "image",
				"title":  "Sample Image",
				"format": "jpeg",
			},
		},
	}

	for _, item := range vectors {
		err := db.Insert(item.id, item.vector, item.metadata)
		if err != nil {
			log.Fatalf("Failed to insert vector %s: %v", item.id, err)
		}
		fmt.Printf("  Inserted vector: %s\n", item.id)
	}

	fmt.Printf("Insertion complete. Database length: %d\n", db.Len())

	// Retrieve a specific vector
	fmt.Println("\n3. Retrieving a specific vector...")
	vector, metadata, err := db.Get("doc1")
	if err != nil {
		log.Fatalf("Failed to get vector: %v", err)
	}

	fmt.Printf("Retrieved vector 'doc1':\n")
	fmt.Printf("  Vector: %v\n", vector)
	fmt.Printf("  Metadata: %v\n", metadata)

	// Perform similarity search
	fmt.Println("\n4. Performing similarity search...")
	queryVector := veclite.Vector{1.0, 0.1, 0.1} // Similar to doc1
	results, err := db.Search(queryVector, 3)
	if err != nil {
		log.Fatalf("Search failed: %v", err)
	}

	fmt.Printf("Search results for query [1.0, 0.1, 0.1] (top 3):\n")
	for i, result := range results {
		fmt.Printf("  %d. ID: %s, Score: %.4f, Type: %s, Title: %s\n",
			i+1, result.ID, result.Score,
			result.Metadata["type"], result.Metadata["title"])
	}

	// Show available distance metrics
	fmt.Println("\n5. Available distance metrics...")
	metrics, err := veclite.GetAvailableMetrics()
	if err != nil {
		log.Fatalf("Failed to get available metrics: %v", err)
	}

	fmt.Printf("Available distance metrics: %v\n", metrics)

	// Save database to file
	fmt.Println("\n6. Saving database to file...")
	dbPath := "example.vlt"
	err = db.Save(dbPath)
	if err != nil {
		log.Fatalf("Failed to save database: %v", err)
	}
	fmt.Printf("Database saved to: %s\n", dbPath)

	// Close current database
	db.Close()

	// Load database from file
	fmt.Println("\n7. Loading database from file...")
	loadedDB, err := veclite.Load(dbPath)
	if err != nil {
		log.Fatalf("Failed to load database: %v", err)
	}
	defer loadedDB.Close()

	fmt.Printf("Database loaded successfully. Length: %d\n", loadedDB.Len())

	// Verify loaded data with a search
	fmt.Println("\n8. Verifying loaded data with search...")
	verificationResults, err := loadedDB.Search(queryVector, 2)
	if err != nil {
		log.Fatalf("Verification search failed: %v", err)
	}

	fmt.Printf("Verification search results:\n")
	for i, result := range verificationResults {
		fmt.Printf("  %d. ID: %s, Score: %.4f\n", i+1, result.ID, result.Score)
	}

	// Delete a vector
	fmt.Println("\n9. Deleting a vector...")
	err = loadedDB.Delete("doc3")
	if err != nil {
		log.Fatalf("Failed to delete vector: %v", err)
	}
	fmt.Printf("Vector 'doc3' deleted. New length: %d\n", loadedDB.Len())

	// Try to get the deleted vector (should fail)
	fmt.Println("\n10. Attempting to retrieve deleted vector...")
	_, _, err = loadedDB.Get("doc3")
	if err != nil {
		fmt.Printf("Expected error retrieving deleted vector: %v\n", err)
	} else {
		fmt.Println("Unexpected: deleted vector was still found!")
	}

	fmt.Println("\n✅ VecLite Go bindings example completed successfully!")
}