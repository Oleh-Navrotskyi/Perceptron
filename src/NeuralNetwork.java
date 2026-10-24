import java.io.Serializable;
import java.util.Random;

public class NeuralNetwork implements Serializable {

    // Параметри
    private final int INPUT_NODES = 784;
    private final int HIDDEN_NODES_1 = 16;
    private final int HIDDEN_NODES_2 = 16;
    private final int OUTPUT_NODES = 10;
    private double learningRate = 0.01;

    // Ваги та зміщення
    private Matrix weights_input_h1, weights_h1_h2, weights_h2_output;
    private Matrix bias_h1, bias_h2, bias_output;

    public NeuralNetwork() {
        weights_input_h1 = new Matrix(HIDDEN_NODES_1, INPUT_NODES);
        weights_h1_h2 = new Matrix(HIDDEN_NODES_2, HIDDEN_NODES_1);
        weights_h2_output = new Matrix(OUTPUT_NODES, HIDDEN_NODES_2);

        bias_h1 = new Matrix(HIDDEN_NODES_1, 1);
        bias_h2 = new Matrix(HIDDEN_NODES_2, 1);
        bias_output = new Matrix(OUTPUT_NODES, 1);

        initializeRandom(weights_input_h1);
        initializeRandom(weights_h1_h2);
        initializeRandom(weights_h2_output);
        initializeRandom(bias_h1);
        initializeRandom(bias_h2);
        initializeRandom(bias_output);
    }

    private void initializeRandom(Matrix m) {
        Random rand = new Random();
        for (int i = 0; i < m.rows; i++) {
            for (int j = 0; j < m.cols; j++) {
                m.data[i][j] = rand.nextGaussian() * 0.5; // Трохи зменшуємо розкид
            }
        }
    }

    // Функція активації (Sigmoid)
    private double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // Похідна сигмоїди: σ(x) * (1 - σ(x))
    // Важливо: ми передаємо сюди вже активоване значення y = σ(x)
    private double dsigmoid(double y) {
        return y * (1.0 - y);
    }

    // === FORWARD PROPAGATION (Передбачення) ===
    public double[] predict(double[] inputArr) {
        // 1. Вхід -> Шар 1
        Matrix inputs = Matrix.fromArray(inputArr);

        Matrix hidden1 = weights_input_h1.product(inputs);
        hidden1.add(bias_h1);
        hidden1 = hidden1.map(this::sigmoid); // Активація

        // 2. Шар 1 -> Шар 2
        Matrix hidden2 = weights_h1_h2.product(hidden1);
        hidden2.add(bias_h2);
        hidden2 = hidden2.map(this::sigmoid);

        // 3. Шар 2 -> Вихід
        Matrix output = weights_h2_output.product(hidden2);
        output.add(bias_output);
        output = output.map(this::sigmoid);

        return output.toArray();
    }

    // === BACKPROPAGATION (Навчання) ===
    // targetArr - це One-hot вектор (наприклад, [0, 0, 1, 0...] для цифри 2)
    public void train(double[] inputArr, double[] targetArr) {
        // КРОК 1: Forward Pass (повторюємо логіку predict, але зберігаємо проміжні результати)
        Matrix inputs = Matrix.fromArray(inputArr);

        Matrix hidden1 = weights_input_h1.product(inputs);
        hidden1.add(bias_h1);
        hidden1 = hidden1.map(this::sigmoid);

        Matrix hidden2 = weights_h1_h2.product(hidden1);
        hidden2.add(bias_h2);
        hidden2 = hidden2.map(this::sigmoid);

        Matrix outputs = weights_h2_output.product(hidden2);
        outputs.add(bias_output);
        outputs = outputs.map(this::sigmoid);

        // КРОК 2: Обчислення помилок і градієнтів
        Matrix targets = Matrix.fromArray(targetArr);

        // --- ВИХІДНИЙ ШАР ---
        // Error = Targets - Outputs
        Matrix output_errors = Matrix.subtract(targets, outputs);

        // Gradient = Error * dsigmoid(Output) * lr
        Matrix gradients = outputs.map(this::dsigmoid); // Похідна
        gradients.multiply(output_errors);             // Hadamard product
        gradients.multiply(learningRate);              // Скаляр

        // Deltas (зміни ваг) = Gradient * Hidden2_Transposed
        Matrix hidden2_T = hidden2.transpose();
        Matrix weights_ho_deltas = gradients.product(hidden2_T);

        // Оновлюємо ваги та зміщення (W_new = W_old + Delta)
        weights_h2_output.add(weights_ho_deltas);
        bias_output.add(gradients); // Delta для bias це просто градієнт

        // --- ПРИХОВАНИЙ ШАР 2 ---
        // Error_H2 = W_HO_Transposed * Error_Output
        Matrix who_T = weights_h2_output.transpose();
        Matrix hidden2_errors = who_T.product(output_errors);

        // Gradient H2
        Matrix h2_gradients = hidden2.map(this::dsigmoid);
        h2_gradients.multiply(hidden2_errors);
        h2_gradients.multiply(learningRate);

        // Deltas H1 -> H2
        Matrix hidden1_T = hidden1.transpose();
        Matrix weights_h1h2_deltas = h2_gradients.product(hidden1_T);

        weights_h1_h2.add(weights_h1h2_deltas);
        bias_h2.add(h2_gradients);

        // --- ПРИХОВАНИЙ ШАР 1 ---
        // Error_H1 = W_H1H2_Transposed * Error_H2
        Matrix wh1h2_T = weights_h1_h2.transpose();
        Matrix hidden1_errors = wh1h2_T.product(hidden2_errors);

        // Gradient H1
        Matrix h1_gradients = hidden1.map(this::dsigmoid);
        h1_gradients.multiply(hidden1_errors);
        h1_gradients.multiply(learningRate);

        // Deltas Input -> H1
        Matrix inputs_T = inputs.transpose();
        Matrix weights_ih1_deltas = h1_gradients.product(inputs_T);

        weights_input_h1.add(weights_ih1_deltas);
        bias_h1.add(h1_gradients);
    }
}