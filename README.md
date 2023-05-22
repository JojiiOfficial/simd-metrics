# simd-metrics
A small library for SIMD accelerated vector distances, extracted from the [qdrant](https://github.com/qdrant/qdrant) project

### Changes (to qdrants implementation)
- VectorElementType and ScoreType were replaced with f32
- Distance has been moved to metric.rs and their imports were updated in the other files.
- 'JsonSchema' and 'FromPrimitive' aren't derived for `Distance`
