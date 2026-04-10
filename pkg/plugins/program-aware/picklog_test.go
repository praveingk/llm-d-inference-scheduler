package programaware

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestPickLogger_Close(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "pick.jsonl")
	logger, err := NewPickLogger(tmpFile)
	require.NoError(t, err)

	err = logger.Close()
	assert.NoError(t, err)

	// Double-close should be a no-op.
	err = logger.Close()
	assert.NoError(t, err)
}

func TestPickLogger_CloseFlushesData(t *testing.T) {
	tmpFile := filepath.Join(t.TempDir(), "pick.jsonl")
	logger, err := NewPickLogger(tmpFile)
	require.NoError(t, err)

	entry := PickLogEntry{Strategy: "test", WinnerID: "prog-a"}
	err = logger.Log(entry)
	require.NoError(t, err)

	err = logger.Close()
	require.NoError(t, err)

	data, err := os.ReadFile(tmpFile)
	require.NoError(t, err)
	assert.Contains(t, string(data), "prog-a")
}