use crate::math::Matrix;

/// Linear (fully connected) layer
pub struct Linear {
    weights: Matrix,
    bias: Option<Matrix>,
    input_dim: usize,
    output_dim: usize,
}

impl Linear {
    /// Create a new linear layer
    /// weights are initialized with small random values (Xavier initialization approximation)
    pub fn new(input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        // Simple initialization: small random values between -0.1 and 0.1
        let mut weights = Matrix::new(output_dim, input_dim);
        for i in 0..output_dim {
            for j in 0..input_dim {
                // Simple pseudo-random initialization (not cryptographically secure)
                let val = ((i * 7919 + j * 9973) % 2000) as f32 / 10000.0 - 0.1;
                weights.set(i, j, val);
            }
        }

        let bias = if use_bias {
            Some(Matrix::new(1, output_dim))
        } else {
            None
        };

        Linear {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: output = input * weights^T + bias
    pub fn forward(&self, input: &Matrix) -> Matrix {
        // input: (batch_size, input_dim)
        // weights: (output_dim, input_dim)
        // output: (batch_size, output_dim)
        let output = input.matmul(&self.weights.transpose());
        
        if let Some(ref bias) = self.bias {
            // Broadcast bias to batch
            let mut result = Matrix::new(output.shape().0, output.shape().1);
            for i in 0..result.shape().0 {
                for j in 0..result.shape().1 {
                    result.set(i, j, output.get(i, j) + bias.get(0, j));
                }
            }
            result
        } else {
            output
        }
    }

    pub fn get_weights(&self) -> &Matrix {
        &self.weights
    }

    pub fn get_weights_mut(&mut self) -> &mut Matrix {
        &mut self.weights
    }

    pub fn get_bias(&self) -> Option<&Matrix> {
        self.bias.as_ref()
    }

    pub fn get_bias_mut(&mut self) -> Option<&mut Matrix> {
        self.bias.as_mut()
    }
}
