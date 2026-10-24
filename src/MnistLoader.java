import java.io.*;
import java.util.*;

public class MnistLoader {

    public static class Data {
        public double[] inputs;  // Зображення 784 пікселі (0.0 - 1.0)
        public double[] targets; // Очікуваний вихід (One-hot vector)

        public Data(double[] inputs, double[] targets) {
            this.inputs = inputs;
            this.targets = targets;
        }
    }

    public List<Data> loadData(String imgPath, String labelPath) throws IOException {
        List<Data> dataset = new ArrayList<>();

        try (DataInputStream imgIn = new DataInputStream(new FileInputStream(imgPath));
             DataInputStream labelIn = new DataInputStream(new FileInputStream(labelPath))) {

            // 1. Читаємо заголовки (Magic numbers та розміри)
            int imgMagic = imgIn.readInt();
            int labelMagic = labelIn.readInt();

            if (imgMagic != 2051 || labelMagic != 2049) {
                throw new IOException("Неправильні файли MNIST!");
            }

            int numImages = imgIn.readInt();
            int numLabels = labelIn.readInt();
            int rows = imgIn.readInt();
            int cols = imgIn.readInt();
            int size = rows * cols; // 28 * 28 = 784
            System.out.println("Починаємо зчитувати дані");
            // 2. Читаємо дані зображення за зображенням
            for (int i = 0; i < numImages; i++) {
                // Читаємо пікселі
                double[] input = new double[size];
                for (int j = 0; j < size; j++) {
                    // Java byte signed (-128..127), а нам треба unsigned (0..255).
                    // Тому робимо побітове AND з 0xFF.
                    int pixel = imgIn.readUnsignedByte();
                    input[j] = pixel / 255.0; // Нормалізація 0..1
                }

                // Читаємо лейбл (цифра 0-9)
                int label = labelIn.readUnsignedByte();
                double[] target = new double[10];
                target[label] = 1.0; // One-hot encoding (наприклад, для 3: [0,0,0,1,0...])

                dataset.add(new Data(input, target));
            }
        }
        System.out.println("Data was read");
        return dataset;
    }
}