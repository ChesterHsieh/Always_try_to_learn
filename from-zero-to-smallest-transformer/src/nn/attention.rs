use crate::math::Matrix;
use crate::math::activation::softmax;
use crate::nn::linear::Linear;

/// Multi-head self-attention mechanism
pub struct MultiHeadAttention {
    num_heads: usize,
    head_dim: usize,
    embed_dim: usize,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    scale: f32,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer
    pub fn new(embed_dim: usize, num_heads: usize) -> Self {
        assert!(embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads");
        
        let head_dim = embed_dim / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        MultiHeadAttention {
            num_heads,
            head_dim,
            embed_dim,
            q_proj: Linear::new(embed_dim, embed_dim, false),
            k_proj: Linear::new(embed_dim, embed_dim, false),
            v_proj: Linear::new(embed_dim, embed_dim, false),
            out_proj: Linear::new(embed_dim, embed_dim, true),
            scale,
        }
    }

    /// Scaled dot-product attention
    /// query, key, value: (seq_len, embed_dim)
    fn scaled_dot_product_attention(
        &self,
        query: &Matrix,
        key: &Matrix,
        value: &Matrix,
    ) -> Matrix {
        // Compute attention scores: Q * K^T / sqrt(d_k)
        let scores = query.matmul(&key.transpose()).scale(self.scale);
        
        // Apply softmax to get attention weights
        let seq_len = scores.shape().0;
        let mut attention_weights = Matrix::new(seq_len, seq_len);
        for i in 0..seq_len {
            let row: Vec<f32> = (0..seq_len).map(|j| scores.get(i, j)).collect();
            let softmax_row = softmax(&row);
            for j in 0..seq_len {
                attention_weights.set(i, j, softmax_row[j]);
            }
        }
        
        // Apply attention weights to values
        attention_weights.matmul(value)
    }

    /// Split matrix into multiple heads
    fn split_heads(&self, x: &Matrix, batch_size: usize) -> Vec<Matrix> {
        let mut heads = Vec::new();
        for h in 0..self.num_heads {
            let start_col = h * self.head_dim;
            
            let mut head = Matrix::new(batch_size, self.head_dim);
            for i in 0..batch_size {
                for j in 0..self.head_dim {
                    head.set(i, j, x.get(i, start_col + j));
                }
            }
            heads.push(head);
        }
        heads
    }

    /// Concatenate heads back together
    fn concat_heads(&self, heads: &[Matrix]) -> Matrix {
        let batch_size = heads[0].shape().0;
        let mut output = Matrix::new(batch_size, self.embed_dim);
        
        for h in 0..self.num_heads {
            let start_col = h * self.head_dim;
            for i in 0..batch_size {
                for j in 0..self.head_dim {
                    output.set(i, start_col + j, heads[h].get(i, j));
                }
            }
        }
        output
    }

    /// Forward pass
    /// input: (batch_size, seq_len, embed_dim) or (seq_len, embed_dim)
    pub fn forward(&self, input: &Matrix) -> Matrix {
        // For simplicity, assume input is (seq_len, embed_dim) - single sequence
        let seq_len = input.shape().0;
        
        // Project to Q, K, V
        let q = self.q_proj.forward(input);
        let k = self.k_proj.forward(input);
        let v = self.v_proj.forward(input);
        
        // Split into heads
        let q_heads = self.split_heads(&q, seq_len);
        let k_heads = self.split_heads(&k, seq_len);
        let v_heads = self.split_heads(&v, seq_len);
        
        // Apply attention for each head
        let mut attended_heads = Vec::new();
        for h in 0..self.num_heads {
            let attended = self.scaled_dot_product_attention(&q_heads[h], &k_heads[h], &v_heads[h]);
            attended_heads.push(attended);
        }
        
        // Concatenate heads
        let concat = self.concat_heads(&attended_heads);
        
        // Output projection
        self.out_proj.forward(&concat)
    }
}
