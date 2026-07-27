# Async Audio Programming

This tutorial covers asynchronous audio programming with coremusic, enabling
non-blocking I/O for responsive applications.

## Why Async?

Asynchronous programming is valuable for audio applications because:

- **Responsive UIs**: File operations don't freeze your interface
- **Concurrent Processing**: Process multiple files simultaneously
- **Server Applications**: Handle multiple clients efficiently
- **Real-time Integration**: Combine audio I/O with network operations

## Prerequisites

- Python 3.11+ (for best async support)
- Basic understanding of Python's `async`/`await` syntax
- Familiarity with coremusic's synchronous API

## Async File Operations

### Basic Async File Reading

Use `AsyncAudioFile` for non-blocking file operations:

```python
--8<-- "examples/tutorials/async_audio/reading.py:info"
```

### Streaming Audio Chunks

Process large files efficiently using async iteration:

```python
--8<-- "examples/tutorials/async_audio/reading.py:chunks"
```

### Concurrent File Processing

Process multiple files simultaneously:

```python
--8<-- "examples/tutorials/async_audio/reading.py:concurrent"
```

## Async Audio Queue

### Basic Async Playback

Use `AsyncAudioQueue` for non-blocking playback control:

```python
--8<-- "examples/tutorials/async_audio/playback.py:queue"
```

### Combining with Other Async Operations

Integrate audio with other async tasks:

```python
--8<-- "examples/tutorials/async_audio/playback.py:monitor"
```

## Error Handling in Async Code

Proper async error handling:

```python
--8<-- "examples/tutorials/async_audio/patterns.py:errors"
```

## Patterns and Best Practices

### Batch Processing Pattern

Efficiently process files in batches:

```python
--8<-- "examples/tutorials/async_audio/patterns.py:batches"
```

### Semaphore Pattern

Limit concurrent operations:

```python
--8<-- "examples/tutorials/async_audio/patterns.py:semaphore"
```

### Producer-Consumer Pattern

For streaming audio processing:

```python
--8<-- "examples/tutorials/async_audio/patterns.py:producer-consumer"
```

## Integration with Web Frameworks

### FastAPI Example

Using coremusic with FastAPI:

```python
import os
import tempfile

from fastapi import FastAPI, UploadFile

from coremusic.audio import AsyncAudioFile

app = FastAPI()

@app.post("/analyze")
async def analyze_audio(file: UploadFile):
    """Analyze uploaded audio file."""
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        async with AsyncAudioFile(tmp_path) as audio:
            return {
                "filename": file.filename,
                "duration": audio.duration,
                "sample_rate": audio.format.sample_rate,
                "channels": audio.format.channels_per_frame
            }
    finally:
        os.unlink(tmp_path)
```

## See Also

- [Audio File Basics](audio_file_basics.md) - Synchronous file operations
- [File Operations Cookbook](../cookbook/file_operations.md) - File operation recipes
- [API Reference](../api/index.md) - API reference
