/// Activation functions for neural networks

/// ReLU activation function: max(0, x)
pub fn relu(x: f32) -> f32 {
    if x > 0.0 {
        x
    } else {
        0.0
    }
}

/// ReLU derivative
pub fn relu_derivative(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

/// GELU activation function: x * 0.5 * (1 + erf(x / sqrt(2)))
/// Using approximation: x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
pub fn gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608;
    let coeff = 0.044715;
    x * 0.5 * (1.0 + tanh_approx(sqrt_2_over_pi * (x + coeff * x * x * x)))
}

/// GELU derivative (approximation)
pub fn gelu_derivative(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608;
    let coeff = 0.044715;
    let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    let tanh_val = tanh_approx(inner);
    0.5 * (1.0 + tanh_val) + x * 0.5 * (1.0 - tanh_val * tanh_val) * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x * x)
}

/// Softmax function for a vector
pub fn softmax(input: &[f32]) -> Vec<f32> {
    // Find max for numerical stability
    let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let exp_sum: f32 = input.iter().map(|x| (x - max_val).exp()).sum();
    input.iter().map(|x| (x - max_val).exp() / exp_sum).collect()
}

/// Tanh approximation using standard library
fn tanh_approx(x: f32) -> f32 {
    let exp_2x = (2.0 * x).exp();
    (exp_2x - 1.0) / (exp_2x + 1.0)
}

/// Sigmoid function
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Sigmoid derivative
pub fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}
