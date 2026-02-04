use crate::math::Matrix;

/// Cross-entropy loss for classification tasks
pub struct CrossEntropyLoss;

impl CrossEntropyLoss {
    /// Compute cross-entropy loss
    /// predictions: (batch_size, num_classes) - logits
    /// targets: (batch_size,) - class indices
    pub fn compute(&self, predictions: &Matrix, targets: &[usize]) -> f32 {
        let batch_size = predictions.shape().0;
        let mut loss = 0.0;
        
        for i in 0..batch_size {
            // Find max for numerical stability
            let max_logit = (0..predictions.shape().1)
                .map(|j| predictions.get(i, j))
                .fold(f32::NEG_INFINITY, f32::max);
            
            // Compute log-sum-exp
            let exp_sum: f32 = (0..predictions.shape().1)
                .map(|j| (predictions.get(i, j) - max_logit).exp())
                .sum();
            
            let log_sum_exp = max_logit + exp_sum.ln();
            
            // Cross-entropy: -log(softmax(target))
            let target_logit = predictions.get(i, targets[i]);
            loss += log_sum_exp - target_logit;
        }
        
        loss / batch_size as f32
    }

    /// Compute gradient of cross-entropy loss w.r.t. predictions
    pub fn backward(&self, predictions: &Matrix, targets: &[usize]) -> Matrix {
        let batch_size = predictions.shape().0;
        let num_classes = predictions.shape().1;
        let mut grad = Matrix::new(batch_size, num_classes);
        
        for i in 0..batch_size {
            // Compute softmax
            let max_logit = (0..num_classes)
                .map(|j| predictions.get(i, j))
                .fold(f32::NEG_INFINITY, f32::max);
            
            let exp_sum: f32 = (0..num_classes)
                .map(|j| (predictions.get(i, j) - max_logit).exp())
                .sum();
            
            // Gradient: softmax - one_hot(target)
            for j in 0..num_classes {
                let softmax_val = (predictions.get(i, j) - max_logit).exp() / exp_sum;
                let one_hot = if j == targets[i] { 1.0 } else { 0.0 };
                grad.set(i, j, softmax_val - one_hot);
            }
        }
        
        grad.scale(1.0 / batch_size as f32)
    }
}

/// Mean Squared Error loss for regression tasks
pub struct MSELoss;

impl MSELoss {
    pub fn compute(&self, predictions: &Matrix, targets: &Matrix) -> f32 {
        assert_eq!(predictions.shape(), targets.shape(), "Predictions and targets must have same shape");
        
        let mut loss = 0.0;
        let total_elements = predictions.shape().0 * predictions.shape().1;
        
        for i in 0..predictions.shape().0 {
            for j in 0..predictions.shape().1 {
                let diff = predictions.get(i, j) - targets.get(i, j);
                loss += diff * diff;
            }
        }
        
        loss / total_elements as f32
    }

    pub fn backward(&self, predictions: &Matrix, targets: &Matrix) -> Matrix {
        assert_eq!(predictions.shape(), targets.shape(), "Predictions and targets must have same shape");
        
        let mut grad = Matrix::new(predictions.shape().0, predictions.shape().1);
        let total_elements = predictions.shape().0 * predictions.shape().1;
        
        for i in 0..predictions.shape().0 {
            for j in 0..predictions.shape().1 {
                let diff = predictions.get(i, j) - targets.get(i, j);
                grad.set(i, j, 2.0 * diff / total_elements as f32);
            }
        }
        
        grad
    }
}
