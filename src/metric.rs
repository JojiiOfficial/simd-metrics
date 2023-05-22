// - VectorElementType and ScoreType were replaced by f32 and their imports removed.
// - Additionally Deserialize and Serialize have been imported from serde.
use serde::{Deserialize, Serialize};

/// Defines how to compare vectors
pub trait Metric {
    fn distance() -> Distance;

    /// Greater the value - closer the vectors
    fn similarity(v1: &[f32], v2: &[f32]) -> f32;

    /// Necessary vector transformations performed before adding it to the collection (like normalization)
    /// Return None if metric does not required preprocessing
    fn preprocess(vector: &[f32]) -> Option<Vec<f32>>;

    /// correct metric score for displaying
    fn postprocess(score: f32) -> f32;
}

// Copied from lib/segment/src/types.rs and 'JsonSchema' and 'FromPrimitive' removed from derive
/// Type of internal tags, build from payload
#[derive(Debug, Deserialize, Serialize, Clone, Copy, PartialEq, Eq, Hash)]
/// Distance function types used to compare vectors
pub enum Distance {
    // <https://en.wikipedia.org/wiki/Cosine_similarity>
    Cosine,
    // <https://en.wikipedia.org/wiki/Euclidean_distance>
    Euclid,
    // <https://en.wikipedia.org/wiki/Dot_product>
    Dot,
}
