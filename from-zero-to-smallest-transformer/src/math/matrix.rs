/// Matrix implementation for neural network operations
/// All operations are done using only Rust standard library
pub struct Matrix {
    data: Vec<Vec<f32>>,
    rows: usize,
    cols: usize,
}

impl Matrix {
    /// Create a new matrix with given dimensions, initialized to zeros
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            data: vec![vec![0.0; cols]; rows],
            rows,
            cols,
        }
    }

    /// Create a matrix from a 2D vector
    pub fn from_vec(data: Vec<Vec<f32>>) -> Self {
        let rows = data.len();
        let cols = if rows > 0 { data[0].len() } else { 0 };
        Matrix { data, rows, cols }
    }

    /// Get dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    /// Get element at position (i, j)
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[i][j]
    }

    /// Set element at position (i, j)
    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        self.data[i][j] = value;
    }

    /// Matrix multiplication: self * other
    pub fn matmul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "Matrix dimensions mismatch for multiplication");
        
        let mut result = Matrix::new(self.rows, other.cols);
        
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        
        result
    }

    /// Element-wise addition
    pub fn add(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match for addition");
        
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] + other.data[i][j];
            }
        }
        result
    }

    /// Element-wise multiplication
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.shape(), other.shape(), "Matrix dimensions must match for multiplication");
        
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * other.data[i][j];
            }
        }
        result
    }

    /// Scalar multiplication
    pub fn scale(&self, scalar: f32) -> Matrix {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = self.data[i][j] * scalar;
            }
        }
        result
    }

    /// Transpose matrix
    pub fn transpose(&self) -> Matrix {
        let mut result = Matrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[j][i] = self.data[i][j];
            }
        }
        result
    }

    /// Apply function element-wise
    pub fn map<F>(&self, f: F) -> Matrix
    where
        F: Fn(f32) -> f32,
    {
        let mut result = Matrix::new(self.rows, self.cols);
        for i in 0..self.rows {
            for j in 0..self.cols {
                result.data[i][j] = f(self.data[i][j]);
            }
        }
        result
    }

    /// Get reference to internal data (for cloning)
    pub fn data(&self) -> &Vec<Vec<f32>> {
        &self.data
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Matrix::from_vec(self.data.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let m = Matrix::new(3, 4);
        assert_eq!(m.shape(), (3, 4));
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Matrix::from_vec(vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ]);
        let b = Matrix::from_vec(vec![
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ]);
        let c = a.matmul(&b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
    }
}
