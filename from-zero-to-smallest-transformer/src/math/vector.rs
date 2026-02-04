/// Vector operations for neural networks
pub struct Vector {
    data: Vec<f32>,
}

impl Vector {
    pub fn new(size: usize) -> Self {
        Vector {
            data: vec![0.0; size],
        }
    }

    pub fn from_vec(data: Vec<f32>) -> Self {
        Vector { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn get(&self, i: usize) -> f32 {
        self.data[i]
    }

    pub fn set(&mut self, i: usize, value: f32) {
        self.data[i] = value;
    }

    pub fn dot(&self, other: &Vector) -> f32 {
        assert_eq!(self.len(), other.len(), "Vectors must have same length for dot product");
        self.data.iter().zip(other.data.iter()).map(|(a, b)| a * b).sum()
    }

    pub fn add(&self, other: &Vector) -> Vector {
        assert_eq!(self.len(), other.len(), "Vectors must have same length for addition");
        Vector {
            data: self.data.iter().zip(other.data.iter()).map(|(a, b)| a + b).collect(),
        }
    }

    pub fn scale(&self, scalar: f32) -> Vector {
        Vector {
            data: self.data.iter().map(|x| x * scalar).collect(),
        }
    }
}
