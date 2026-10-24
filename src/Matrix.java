import java.io.Serializable;
import java.util.function.DoubleUnaryOperator;

public class Matrix implements Serializable {
    int rows, cols;
    double[][] data;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        data = new double[rows][cols];
    }

    // Статичний метод для перетворення масиву в вектор-стовпчик
    public static Matrix fromArray(double[] arr) {
        Matrix m = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            m.data[i][0] = arr[i];
        }
        return m;
    }

    // Перетворення назад у масив (для отримання результату)
    public double[] toArray() {
        double[] arr = new double[rows * cols];
        int k = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                arr[k++] = data[i][j];
            }
        }
        return arr;
    }

    // Матричне множення (Dot Product)
    public Matrix product(Matrix m) {
        if (this.cols != m.rows) {
            throw new IllegalArgumentException("Розміри матриць не підходять для множення!");
        }
        Matrix result = new Matrix(this.rows, m.cols);
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                double sum = 0;
                for (int k = 0; k < this.cols; k++) { // k йде по ширині першої матриці
                    sum += this.data[i][k] * m.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }

    // Транспонування (повертає НОВУ матрицю)
    public Matrix transpose() {
        Matrix result = new Matrix(this.cols, this.rows); // Розміри перевернуті
        for (int i = 0; i < this.rows; i++) {
            for (int j = 0; j < this.cols; j++) {
                result.data[j][i] = this.data[i][j];
            }
        }
        return result;
    }

    // Додавання матриць (Element-wise)
    public void add(Matrix m) {
        // Тут можна додати перевірку розмірів
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] += m.data[i][j];
            }
        }
    }

    // Віднімання (Element-wise): this - m
    // Повертає нову матрицю, щоб не псувати поточні дані при обчисленні помилок
    public static Matrix subtract(Matrix a, Matrix b) {
        Matrix result = new Matrix(a.rows, a.cols);
        for (int i = 0; i < a.rows; i++) {
            for (int j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }
        return result;
    }

    // Поелементне множення (Hadamard product) - потрібно для бекпропагації
    // A ⊙ B
    public void multiply(Matrix m) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] *= m.data[i][j];
            }
        }
    }

    // Множення на скаляр (Learning Rate)
    public void multiply(double x) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] *= x;
            }
        }
    }

    // Застосування функції (Sigmoid, Derivative)
    public Matrix map(DoubleUnaryOperator func) {
        Matrix result = new Matrix(rows, cols); // Повертаємо нову, щоб мати і оригінал, і результат
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func.applyAsDouble(this.data[i][j]);
            }
        }
        return result;
    }
}
