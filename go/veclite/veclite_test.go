package veclite

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestVecLiteNew(t *testing.T) {
	db, err := New()
	require.NoError(t, err)
	require.NotNil(t, db)

	assert.True(t, db.IsEmpty())
	assert.Equal(t, uint(0), db.Len())

	err = db.Close()
	assert.NoError(t, err)
}

func TestVecLiteNewWithConfig(t *testing.T) {
	config := &Config{
		MaxVectors: 1000,
		DefaultK:   5,
	}

	db, err := NewWithConfig(config)
	require.NoError(t, err)
	require.NotNil(t, db)

	assert.True(t, db.IsEmpty())
	assert.Equal(t, uint(0), db.Len())

	err = db.Close()
	assert.NoError(t, err)
}

func TestVecLiteInsertAndGet(t *testing.T) {
	db, err := New()
	require.NoError(t, err)
	defer db.Close()

	// Test data
	id := "test_vector"
	vector := Vector{1.0, 2.0, 3.0}
	metadata := Metadata{
		"type":        "test",
		"description": "A test vector",
	}

	// Insert vector
	err = db.Insert(id, vector, metadata)
	assert.NoError(t, err)

	// Check database state
	assert.False(t, db.IsEmpty())
	assert.Equal(t, uint(1), db.Len())

	// Get vector
	gotVector, gotMetadata, err := db.Get(id)
	require.NoError(t, err)

	// Verify vector data
	assert.Equal(t, len(vector), len(gotVector))
	for i, v := range vector {
		assert.InDelta(t, v, gotVector[i], 0.001)
	}

	// Verify metadata
	assert.Equal(t, metadata, gotMetadata)
}

func TestVecLiteDelete(t *testing.T) {
	db, err := New()
	require.NoError(t, err)
	defer db.Close()

	// Insert a vector
	id := "delete_me"
	vector := Vector{1.0, 2.0, 3.0}
	metadata := Metadata{"test": "delete"}

	err = db.Insert(id, vector, metadata)
	assert.NoError(t, err)
	assert.Equal(t, uint(1), db.Len())

	// Delete the vector
	err = db.Delete(id)
	assert.NoError(t, err)

	// Verify deletion
	assert.True(t, db.IsEmpty())
	assert.Equal(t, uint(0), db.Len())

	// Try to get deleted vector (should fail)
	_, _, err = db.Get(id)
	assert.Error(t, err)
}

func TestVecLiteSearch(t *testing.T) {
	db, err := New()
	require.NoError(t, err)
	defer db.Close()

	// Insert test vectors
	vectors := []struct {
		id       string
		vector   Vector
		metadata Metadata
	}{
		{"vec1", Vector{1.0, 0.0, 0.0}, Metadata{"type": "A"}},
		{"vec2", Vector{0.0, 1.0, 0.0}, Metadata{"type": "B"}},
		{"vec3", Vector{0.0, 0.0, 1.0}, Metadata{"type": "A"}},
		{"vec4", Vector{1.0, 1.0, 0.0}, Metadata{"type": "C"}},
	}

	for _, v := range vectors {
		err := db.Insert(v.id, v.vector, v.metadata)
		assert.NoError(t, err)
	}

	// Search for similar vectors
	query := Vector{1.0, 0.1, 0.1} // Similar to vec1
	results, err := db.Search(query, 2)
	require.NoError(t, err)

	// Should return 2 results
	assert.Len(t, results, 2)

	// First result should be vec1 (most similar)
	assert.Equal(t, "vec1", results[0].ID)
	assert.Equal(t, "A", results[0].Metadata["type"])

	// Results should be sorted by score (ascending for distance)
	assert.True(t, results[0].Score <= results[1].Score)
}

func TestVecLiteSaveAndLoad(t *testing.T) {
	// Create temporary file
	tempDir := t.TempDir()
	dbPath := filepath.Join(tempDir, "test.vlt")

	// Create and populate original database
	originalDB, err := New()
	require.NoError(t, err)

	testData := []struct {
		id       string
		vector   Vector
		metadata Metadata
	}{
		{"doc1", Vector{1.0, 2.0, 3.0}, Metadata{"type": "document", "source": "test"}},
		{"doc2", Vector{4.0, 5.0, 6.0}, Metadata{"type": "image", "format": "png"}},
		{"doc3", Vector{7.0, 8.0, 9.0}, Metadata{"type": "document", "language": "en"}},
	}

	for _, item := range testData {
		err := originalDB.Insert(item.id, item.vector, item.metadata)
		require.NoError(t, err)
	}

	originalLen := originalDB.Len()

	// Save database
	err = originalDB.Save(dbPath)
	assert.NoError(t, err)

	// Check that file was created
	_, err = os.Stat(dbPath)
	assert.NoError(t, err)

	originalDB.Close()

	// Load database
	loadedDB, err := Load(dbPath)
	require.NoError(t, err)
	defer loadedDB.Close()

	// Verify loaded data
	assert.Equal(t, originalLen, loadedDB.Len())
	assert.False(t, loadedDB.IsEmpty())

	// Verify all vectors can be retrieved
	for _, item := range testData {
		gotVector, gotMetadata, err := loadedDB.Get(item.id)
		require.NoError(t, err, "Failed to get vector %s", item.id)

		// Check vector
		assert.Equal(t, len(item.vector), len(gotVector))
		for i, expected := range item.vector {
			assert.InDelta(t, expected, gotVector[i], 0.001)
		}

		// Check metadata
		assert.Equal(t, item.metadata, gotMetadata)
	}

	// Test search on loaded database
	query := Vector{1.1, 2.1, 3.1}
	results, err := loadedDB.Search(query, 2)
	require.NoError(t, err)
	assert.Len(t, results, 2)
}

func TestGetAvailableMetrics(t *testing.T) {
	metrics, err := GetAvailableMetrics()
	require.NoError(t, err)
	require.NotEmpty(t, metrics)

	// Should contain standard distance metrics
	expectedMetrics := []string{"euclidean", "cosine", "dot_product", "manhattan"}
	for _, expected := range expectedMetrics {
		assert.Contains(t, metrics, expected)
	}
}

func TestErrorHandling(t *testing.T) {
	db, err := New()
	require.NoError(t, err)
	defer db.Close()

	// Test getting non-existent vector
	_, _, err = db.Get("nonexistent")
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrVectorNotFound)

	// Test deleting non-existent vector
	err = db.Delete("nonexistent")
	assert.Error(t, err)
	assert.ErrorIs(t, err, ErrVectorNotFound)

	// Test invalid file path for loading
	_, err = Load("/invalid/path/that/does/not/exist.vlt")
	assert.Error(t, err)
}

func TestMemoryManagement(t *testing.T) {
	// Test that multiple databases can be created and closed
	databases := make([]*VecLite, 10)

	for i := 0; i < 10; i++ {
		db, err := New()
		require.NoError(t, err)
		databases[i] = db

		// Insert a vector in each database
		err = db.Insert("test", Vector{float32(i), float32(i + 1), float32(i + 2)}, Metadata{"index": string(rune(i + '0'))})
		assert.NoError(t, err)
	}

	// Close all databases
	for _, db := range databases {
		err := db.Close()
		assert.NoError(t, err)
	}
}

func TestVectorDimensions(t *testing.T) {
	db, err := New()
	require.NoError(t, err)
	defer db.Close()

	// Insert first vector with 3 dimensions
	err = db.Insert("vec1", Vector{1.0, 2.0, 3.0}, Metadata{})
	assert.NoError(t, err)

	// Try to insert vector with different dimensions (should fail)
	err = db.Insert("vec2", Vector{1.0, 2.0}, Metadata{})
	assert.Error(t, err)

	// Try to search with wrong dimensions
	_, err = db.Search(Vector{1.0, 2.0}, 1)
	assert.Error(t, err)
}

func TestLargeDataset(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping large dataset test in short mode")
	}

	db, err := New()
	require.NoError(t, err)
	defer db.Close()

	// Insert 1000 vectors
	vectorCount := 1000
	dimensions := 128

	for i := 0; i < vectorCount; i++ {
		vector := make(Vector, dimensions)
		for j := 0; j < dimensions; j++ {
			vector[j] = float32(i*dimensions+j) / 1000.0
		}

		metadata := Metadata{
			"index": string(rune(i)),
			"type":  "synthetic",
		}

		err := db.Insert(fmt.Sprintf("vec_%d", i), vector, metadata)
		assert.NoError(t, err)
	}

	assert.Equal(t, uint(vectorCount), db.Len())

	// Perform search
	query := make(Vector, dimensions)
	for j := 0; j < dimensions; j++ {
		query[j] = float32(j) / 1000.0
	}

	results, err := db.Search(query, 10)
	require.NoError(t, err)
	assert.Len(t, results, 10)

	// Verify results are sorted by score
	for i := 1; i < len(results); i++ {
		assert.True(t, results[i-1].Score <= results[i].Score,
			"Results should be sorted by score (ascending)")
	}
}

func BenchmarkVecLiteInsert(b *testing.B) {
	db, err := New()
	require.NoError(b, err)
	defer db.Close()

	vector := Vector{1.0, 2.0, 3.0, 4.0, 5.0}
	metadata := Metadata{"benchmark": "insert"}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		id := fmt.Sprintf("vec_%d", i)
		err := db.Insert(id, vector, metadata)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkVecLiteSearch(b *testing.B) {
	db, err := New()
	require.NoError(b, err)
	defer db.Close()

	// Pre-populate database
	for i := 0; i < 1000; i++ {
		vector := Vector{
			float32(i),
			float32(i + 1),
			float32(i + 2),
			float32(i + 3),
			float32(i + 4),
		}
		metadata := Metadata{"benchmark": "search"}
		err := db.Insert(fmt.Sprintf("vec_%d", i), vector, metadata)
		require.NoError(b, err)
	}

	query := Vector{1.0, 2.0, 3.0, 4.0, 5.0}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := db.Search(query, 10)
		if err != nil {
			b.Fatal(err)
		}
	}
}