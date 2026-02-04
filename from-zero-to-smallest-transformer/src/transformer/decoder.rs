use crate::math::Matrix;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::layer_norm::LayerNorm;
use crate::nn::feed_forward::FeedForward;

/// Transformer decoder block
pub struct DecoderBlock {
    self_attn: MultiHeadAttention,
    cross_attn: MultiHeadAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
    ff: FeedForward,
}

impl DecoderBlock {
    pub fn new(embed_dim: usize, num_heads: usize, ff_dim: usize) -> Self {
        DecoderBlock {
            self_attn: MultiHeadAttention::new(embed_dim, num_heads),
            cross_attn: MultiHeadAttention::new(embed_dim, num_heads),
            norm1: LayerNorm::new(embed_dim),
            norm2: LayerNorm::new(embed_dim),
            norm3: LayerNorm::new(embed_dim),
            ff: FeedForward::new(embed_dim, ff_dim),
        }
    }

    /// Forward pass
    /// x: decoder input (seq_len, embed_dim)
    /// encoder_out: encoder output (seq_len, embed_dim)
    pub fn forward(&self, x: &Matrix, encoder_out: &Matrix) -> Matrix {
        // Self-attention
        let self_attn_out = self.self_attn.forward(x);
        let x1 = self_attn_out.add(x);
        let x1_norm = self.norm1.forward(&x1);
        
        // Cross-attention
        let cross_attn_out = self.cross_attn.forward(&x1_norm);
        let x2 = cross_attn_out.add(&x1_norm);
        let x2_norm = self.norm2.forward(&x2);
        
        // Feed-forward
        let ff_out = self.ff.forward(&x2_norm);
        let x3 = ff_out.add(&x2_norm);
        self.norm3.forward(&x3)
    }
}
