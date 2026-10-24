import java.io.*;
import java.util.List;
import java.util.Collections;

public class Main {
    public static void main(String[] args) throws Exception {
        System.out.println("Neural Network Params are loading");
        ObjectInputStream in = new ObjectInputStream(new FileInputStream("first_saved_network.txt"));
        NeuralNetwork nn = (NeuralNetwork) in.readObject();
        in.close();
        System.out.println("Завантаження даних...");
        MnistLoader loader = new MnistLoader();
        // Вкажи тут шляхи до своїх файлів
        List<MnistLoader.Data> trainingData = loader.loadData("train-images.idx3-ubyte", "train-labels.idx1-ubyte");
        System.out.println("Починаємо навчання...");
        int epochs = 15;
        
        for (int i = 0; i < epochs; i++) {
            // Перемішуємо дані перед кожною епохою для кращого навчання
            Collections.shuffle(trainingData);
            
            for (MnistLoader.Data data : trainingData) {
                nn.train(data.inputs, data.targets);
            }
            System.out.println("Епоха " + (i+1) + " завершена.");
            
            // Тут можна додати перевірку точності на тестових даних
        }
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream("first_saved_network.txt"));
        out.writeObject(nn);
        out.close();
        NeuralUI neuralUI = new NeuralUI();
        neuralUI.setOnPredict(()->{
            double [] input = neuralUI.getInputArray();
            neuralUI.updateVisualization(nn.predict(input));
        });
    }
}