# Performance Guide

**Version:** 0.1.8

Best practices, benchmarks, and optimization techniques for achieving optimal performance with CoreMusic.

## Performance Characteristics

### Architecture Overview

CoreMusic uses a hybrid architecture for optimal performance:

```
┌─────────────────────────────────────────────┐
│ Python Layer (High-Level OO API)           │
│ - Convenience and safety                    │
│ - Automatic resource management             │
│ - ~5-10% overhead                           │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│ Cython Layer (capi.pyx)                     │
│ - Minimal Python overhead                   │
│ - Direct C function calls                   │
│ - ~1-2% overhead                            │
└─────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────┐
│ CoreAudio C APIs (Apple Frameworks)         │
│ - Native performance                        │
│ - Hardware-accelerated when available       │
└─────────────────────────────────────────────┘
```

### Performance Tiers

| Operation | API Level | Performance | Use Case |
|-----------|-----------|-------------|----------|
| File I/O | OO API | ~5% overhead | Scripts, prototyping |
| File I/O | Functional API | ~1% overhead | Production pipelines |
| Real-time | Cython callback | Native | Live processing |
| Batch | Parallel utils | Linear scaling | Mass conversion |
| MIDI | OO API | Negligible | Composition tools |

## API Selection

### Choosing the Right API

**Use Object-Oriented API when:**

- Development speed is priority
- Code readability matters
- Automatic cleanup is desired
- Overhead is acceptable (<10%)

**Use Functional API when:**

- Maximum performance is critical
- Processing large files (>100MB)
- Building low-level tools
- Need explicit control

**Use Cython callbacks when:**

- Real-time audio processing
- Custom DSP implementations
- Latency-sensitive operations
- Need to avoid Python GIL

### Performance Comparison

```python
--8<-- "examples/guides/performance/snippet_01.py:example"
```

Expected Results:

```
OO API: 0.0523s
Functional API: 0.0498s
Overhead: 5.0%
```

### Hybrid Approach

Best of both worlds - use OO for convenience, functional for performance:

```python
--8<-- "examples/guides/performance/snippet_02.py:example"
```

## Memory Management

### Resource Lifecycle

**Automatic Cleanup (OO API):**

```python
--8<-- "examples/guides/performance/snippet_03.py:example"
```

**Manual Cleanup (Functional API):**

```python
--8<-- "examples/guides/performance/snippet_04.py:example"
```

### Memory Pooling

Pre-allocate buffers for large operations:

```python
--8<-- "examples/guides/performance/snippet_05.py:example"
```

### Avoiding Memory Leaks

```python
--8<-- "examples/guides/performance/snippet_06.py:example"
```

## Buffer Optimization

### Optimal Buffer Sizes

| Use Case | Buffer Size | Rationale |
|----------|-------------|-----------|
| File I/O | 4096-8192 frames | Balance memory/speed |
| Real-time | 256-512 frames | Low latency |
| Streaming | 8192-16384 | Throughput |
| Batch | 16384-32768 | Maximum speed |

### Buffer Size Tuning

```python
--8<-- "examples/guides/performance/benchmark_buffer_size.py:example"
```

Expected Results:

```
Buffer 512: 12.5 Mframes/sec
Buffer 1024: 18.2 Mframes/sec
Buffer 2048: 22.3 Mframes/sec
Buffer 4096: 24.8 Mframes/sec  <- Sweet spot
Buffer 8192: 25.1 Mframes/sec
Buffer 16384: 25.2 Mframes/sec
```

## Large File Processing

### Chunked Processing

Process large files in manageable chunks:

```python
--8<-- "examples/guides/performance/process_large_file.py:example"
```

### Parallel File Processing

Process multiple files in parallel:

```python
--8<-- "examples/guides/performance/convert_file.py:example"
```

## Real-Time Audio

### Low-Latency Configuration

```python
--8<-- "examples/guides/performance/snippet_10.py:example"
```

### Render Callback Performance

```cython
# Pure Cython callback for maximum performance
# Defined in capi.pyx

cdef OSStatus render_callback(
    void *inRefCon,
    AudioUnitRenderActionFlags *ioActionFlags,
    const AudioTimeStamp *inTimeStamp,
    UInt32 inBusNumber,
    UInt32 inNumberFrames,
    AudioBufferList *ioData
) nogil:
    # No Python overhead
    # No GIL held
    # Direct memory access
    # Native performance

    # Fill audio buffers...
    return 0
```

### Avoiding Dropouts

Best practices for glitch-free real-time audio:

1. **Use appropriate buffer sizes** (256-512 frames)
2. **Minimize allocations** in render callback
3. **Pre-compute** expensive operations
4. **Use lock-free data structures** for communication
5. **Avoid system calls** in callback
6. **Test under load** with other apps running

## Benchmarks

### File I/O Performance

Test: Read 100MB audio file (44.1kHz stereo float32)

| API | Time | Throughput |
|-----|------|------------|
| OO API | 0.423s | 236 MB/s |
| Functional API | 0.401s | 249 MB/s |
| NumPy memmap | 0.387s | 258 MB/s (ref) |

### Format Conversion Performance

Test: Convert 10 minutes of audio (44.1kHz -> 48kHz)

| Method | Time | Speed Ratio |
|--------|------|-------------|
| ExtAudioFile | 2.13s | 282x realtime |
| AudioConverter | 1.98s | 303x realtime |
| SoX (external) | 3.45s | 174x realtime |

### MIDI Processing Performance

Test: Generate 10,000 MIDI notes

| Operation | Time | Notes/sec |
|-----------|------|-----------|
| MusicTrack add | 0.089s | 112,000 |
| Sequence save | 0.142s | 70,000 |
| File load | 0.067s | 149,000 |

### Real-Time Latency

Configuration: 44.1kHz, float32, stereo

| Buffer Size | Latency (ms) | CPU Usage |
|-------------|-------------|-----------|
| 128 frames | 2.9ms | 12% |
| 256 frames | 5.8ms | 6% |
| 512 frames | 11.6ms | 3% |
| 1024 frames | 23.2ms | 2% |

## Profiling and Debugging

### Using Python Profiler

```python
--8<-- "examples/guides/performance/audio_processing_task.py:example"
```

### Memory Profiling

```python
--8<-- "examples/guides/performance/memory_intensive_operation.py:example"
```

### Performance Monitoring

```python
--8<-- "examples/guides/performance/performancemonitor.py:example"
```

## Best Practices Summary

### File I/O

- Use 4096-8192 frame buffers for optimal throughput
- Reuse buffers when processing multiple chunks
- Use ExtendedAudioFile for format conversion
- Close files promptly to release resources

### Real-Time Audio

- Target 256-512 frame buffers for low latency
- Implement render callbacks in Cython for best performance
- Avoid memory allocations in audio thread
- Pre-compute lookup tables and coefficients

### Memory Management

- Always use context managers with OO API
- Dispose objects explicitly when not using context managers
- Pre-allocate buffers for repeated operations
- Use NumPy views instead of copies when possible

### Parallel Processing

- Use ProcessPoolExecutor for CPU-bound tasks
- Divide work into independent chunks
- Use 1-2x CPU cores for optimal scaling
- Monitor memory usage with multiple processes

### API Selection

- Start with OO API for prototyping
- Switch to functional API for bottlenecks
- Use Cython callbacks for real-time code
- Profile before optimizing

## See Also

- [Practical recipes](../cookbook/index.md)
- [API reference](../api/index.md)
- Apple's CoreAudio documentation

!!! note
    Performance characteristics may vary based on:

    - macOS version
    - Hardware specifications
    - Audio format and sample rate
    - System load and background processes
