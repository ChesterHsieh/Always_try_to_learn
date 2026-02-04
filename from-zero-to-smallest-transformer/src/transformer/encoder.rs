use crate::math::Matrix;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::layer_norm::LayerNorm;
use crate::nn::feed_forward::FeedForward;

/// Transformer encoder block
pub struct EncoderBlock {
    self_attn: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    ff: FeedForward,
}

impl EncoderBlock {
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        EncoderBlock {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads),
            norm1: LayerNorm::new(embed_dim),
            norm2: LayerNorm::new(embed_dim),
            ff: FeedForward::new(embed_dim, ff_dim),
        }
    }

    /// Forward pass with residual connections
    /// x: (seq_len, embed_dim)
    pub fn forward(&self, x: &Matrix) -> Matrix {
        // Self-attention with residual and norm
        let attn_out = self.self_attn.forward(x);
        let x1 = attn_out.add(x);  // Residual connection
        let x1_norm = self.norm1.forward(&x1);
        
        // Feed-forward with residual and norm
        let ff_out = self.ff.forward(&x1_norm);
        let x2 = ff_out.add(&x1_norm);  // Residual connection
        self.norm2.forward(&x2)
    }
}
