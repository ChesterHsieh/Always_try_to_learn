use crate::math::Matrix;
use crate::transformer::Transformer;
use crate::training::loss::CrossEntropyLoss;
use crate::training::optimizer::SGD;

/// Trainer for transformer model
pub struct Trainer {
    model: Transformer,
    optimizer: SGD,
    loss_fn: CrossEntropyLoss,
}

impl Trainer {
    pub fn new(model: Transformer, learning_rate: f32) -> Self {
        Trainer {
            model,
            optimizer: SGD::new(learning_rate),
            loss_fn: CrossEntropyLoss,
        }
    }

    /// Train for one epoch
    /// inputs: input sequences (batch_size, seq_len, embed_dim)
    /// targets: target class indices (batch_size,)
    pub fn train_step(&mut self, inputs: &Matrix, targets: &[usize]) -> f32 {
        // Forward pass
        let predictions = self.model.forward(inputs);
        
        // Compute loss
        let loss = self.loss_fn.compute(&predictions, targets);
        
        // Backward pass (gradient computation)
        let grad = self.loss_fn.backward(&predictions, targets);
        
        // TODO: Backpropagate gradients through the model
        // This requires implementing backward() methods for all layers
        
        loss
    }

    /// Evaluate the model
    pub fn evaluate(&self, inputs: &Matrix, targets: &[usize]) -> f32 {
        let predictions = self.model.forward(inputs);
        self.loss_fn.compute(&predictions, targets)
    }
}
