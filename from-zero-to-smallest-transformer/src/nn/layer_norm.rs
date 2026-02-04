use crate::math::Matrix;

/// Layer normalization
pub struct LayerNorm {
    embed_dim: usize,
    gamma: Vec<f32>,  // Scale parameters
    beta: Vec<f32>,  // Shift parameters
    eps: f32,
}

impl LayerNorm {
    pub fn new(embed_dim: usize) -> Self {
        LayerNorm {
            embed_dim,
            gamma: vec![1.0; embed_dim],  // Initialize to 1
            beta: vec![0.0; embed_dim],   // Initialize to 0
            eps: 1e-5,
        }
    }

    /// Normalize along the last dimension
    /// input: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
    pub fn forward(&self, input: &Matrix) -> Matrix {
        let (rows, cols) = input.shape();
        let mut output = Matrix::new(rows, cols);
        
        for i in 0..rows {
            // Compute mean
            let mut mean = 0.0;
            for j in 0..cols {
                mean += input.get(i, j);
            }
            mean /= cols as f32;
            
            // Compute variance
            let mut variance = 0.0;
            for j in 0..cols {
                let diff = input.get(i, j) - mean;
                variance += diff * diff;
            }
            variance /= cols as f32;
            let std = (variance + self.eps).sqrt();
            
            // Normalize and apply affine transformation
            for j in 0..cols {
                let normalized = (input.get(i, j) - mean) / std;
                output.set(i, j, self.gamma[j] * normalized + self.beta[j]);
            }
        }
        
        output
    }
}
