//! RNA velocity on neighbour graphs and its Hodge decomposition.

pub mod flow;
pub mod knn;

pub use flow::edge_flow;
pub use knn::knn_edges;
