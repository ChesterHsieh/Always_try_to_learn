use crate::math::Matrix;
use crate::math::activation::gelu;
use crate::nn::linear::Linear;

/// Feed-forward network (two linear layers with GELU activation)
pub struct FeedForward {
    linear1: Linear,
    linear2: Linear,
}

impl FeedForward {
    pub fn new(embed_dim: usize, ff_dim: usize) -> Self {
        FeedForward {
            linear1: Linear::new(embed_dim, ff_dim, true),
            linear2: Linear::new(ff_dim, embed_dim, true),
        }
    }

    /// Forward pass: linear2(gelu(linear1(x)))
    pub fn forward(&self, input: &Matrix) -> Matrix {
        let x = self.linear1.forward(input);
        let x_gelu = x.map(gelu);
        self.linear2.forward(&x_gelu)
    }
}
