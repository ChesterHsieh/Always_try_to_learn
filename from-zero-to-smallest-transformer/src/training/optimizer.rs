use crate::math::Matrix;

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
}

impl SGD {
    pub fn new(learning_rate: f32) -> Self {
        SGD {
            learning_rate,
            momentum: 0.0,
            weight_decay: 0.0,
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Update weights using gradient descent
    /// weights: mutable reference to weight matrix
    /// gradients: gradient matrix (same shape as weights)
    pub fn update(&self, weights: &mut Matrix, gradients: &Matrix) {
        assert_eq!(weights.shape(), gradients.shape(), "Weights and gradients must have same shape");
        
        for i in 0..weights.shape().0 {
            for j in 0..weights.shape().1 {
                let grad = gradients.get(i, j);
                let weight = weights.get(i, j);
                
                // Apply weight decay
                let grad_with_decay = grad + self.weight_decay * weight;
                
                // Update weight
                let new_weight = weight - self.learning_rate * grad_with_decay;
                weights.set(i, j, new_weight);
            }
        }
    }
}

/// Adam optimizer (simplified version)
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    // State would be stored here in a real implementation
}

impl Adam {
    pub fn new(learning_rate: f32) -> Self {
        Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }

    /// Simplified Adam update (without state tracking)
    /// In a full implementation, you'd track m and v for each parameter
    pub fn update(&self, weights: &mut Matrix, gradients: &Matrix) {
        // Simplified: just use SGD for now
        // Full Adam implementation would require maintaining state
        for i in 0..weights.shape().0 {
            for j in 0..weights.shape().1 {
                let grad = gradients.get(i, j);
                let weight = weights.get(i, j);
                weights.set(i, j, weight - self.learning_rate * grad);
            }
        }
    }
}
