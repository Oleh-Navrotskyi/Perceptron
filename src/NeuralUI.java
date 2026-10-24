import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.image.BufferedImage;
import java.text.DecimalFormat;

public class NeuralUI extends JFrame {

    // Розміри
    private static final int CANVAS_SIZE = 280; // Розмір для малювання
    private static final int MNIST_SIZE = 28;   // Розмір для мережі

    // Компоненти
    private DrawPanel drawPanel;
    private JPanel barsPanel;
    private JProgressBar[] resultBars = new JProgressBar[10];
    private JLabel resultLabel;

    // Колбек для підключення твоєї нейромережі
    private Runnable predictCallback;

    public NeuralUI() {
        super("Neural Network Digit Recognizer");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLayout(new BorderLayout());

        // 1. Панель малювання
        drawPanel = new DrawPanel();
        add(drawPanel, BorderLayout.CENTER);

        // 2. Панель результатів (справа)
        JPanel rightPanel = new JPanel();
        rightPanel.setLayout(new BoxLayout(rightPanel, BoxLayout.Y_AXIS));
        rightPanel.setPreferredSize(new Dimension(200, CANVAS_SIZE));
        rightPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        resultLabel = new JLabel("Draw a digit");
        resultLabel.setFont(new Font("Arial", Font.BOLD, 24));
        resultLabel.setAlignmentX(Component.CENTER_ALIGNMENT);
        rightPanel.add(resultLabel);
        rightPanel.add(Box.createVerticalStrut(20));

        // Стовпчики ймовірностей (0-9)
        for (int i = 0; i < 10; i++) {
            JPanel barContainer = new JPanel(new BorderLayout());
            JLabel label = new JLabel(i + ": ");
            resultBars[i] = new JProgressBar(0, 100);
            resultBars[i].setStringPainted(true);
            
            barContainer.add(label, BorderLayout.WEST);
            barContainer.add(resultBars[i], BorderLayout.CENTER);
            barContainer.setMaximumSize(new Dimension(180, 25));
            rightPanel.add(barContainer);
            rightPanel.add(Box.createVerticalStrut(5));
        }
        add(rightPanel, BorderLayout.EAST);

        // 3. Кнопки (знизу)
        JPanel buttonPanel = new JPanel();
        JButton clearBtn = new JButton("Clear");
        JButton predictBtn = new JButton("Predict");

        clearBtn.addActionListener(e -> {
            drawPanel.clear();
            resetBars();
        });

        predictBtn.addActionListener(e -> {
            if (predictCallback != null) {
                predictCallback.run(); // Викликаємо твій код
            }
        });

        buttonPanel.add(clearBtn);
        buttonPanel.add(predictBtn);
        add(buttonPanel, BorderLayout.SOUTH);

        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    // Метод для передачі логіки передбачення з Main класу
    public void setOnPredict(Runnable action) {
        this.predictCallback = action;
    }

    // Оновлення UI результатами (приймає масив output з нейромережі)
    public void updateVisualization(double[] networkOutput) {
        int maxIndex = 0;
        double maxVal = networkOutput[0];

        for (int i = 0; i < 10; i++) {
            double val = networkOutput[i];
            if (val > maxVal) {
                maxVal = val;
                maxIndex = i;
            }
            // Оновлюємо прогрес-бари
            resultBars[i].setValue((int) (val * 100));
            resultBars[i].setString(new DecimalFormat("#0.0%").format(val));
        }
        resultLabel.setText("Prediction: " + maxIndex);
    }

    private void resetBars() {
        for (JProgressBar bar : resultBars) {
            bar.setValue(0);
            bar.setString("");
        }
        resultLabel.setText("Draw a digit");
    }

    // Головний метод для отримання даних для мережі
    // Повертає масив 784 елементів (0.0 - 1.0)
    public double[] getInputArray() {
        return drawPanel.getMnistData();
    }

    // --- Внутрішній клас для малювання ---
    private class DrawPanel extends JPanel {
        private BufferedImage image;
        private Graphics2D g2;

        public DrawPanel() {
            setPreferredSize(new Dimension(CANVAS_SIZE, CANVAS_SIZE));
            setDoubleBuffered(false);
            
            // Обробка миші
            MouseAdapter mouseHandler = new MouseAdapter() {
                private int prevX, prevY;

                @Override
                public void mousePressed(MouseEvent e) {
                    prevX = e.getX();
                    prevY = e.getY();
                    draw(prevX, prevY, prevX, prevY);
                }

                @Override
                public void mouseDragged(MouseEvent e) {
                    int x = e.getX();
                    int y = e.getY();
                    draw(prevX, prevY, x, y);
                    prevX = x;
                    prevY = y;
                }
            };
            addMouseListener(mouseHandler);
            addMouseMotionListener(mouseHandler);
        }

        @Override
        protected void paintComponent(Graphics g) {
            if (image == null) {
                image = new BufferedImage(CANVAS_SIZE, CANVAS_SIZE, BufferedImage.TYPE_INT_RGB);
                g2 = (Graphics2D) image.getGraphics();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                clear();
            }
            g.drawImage(image, 0, 0, null);
        }

        public void clear() {
            g2.setPaint(Color.BLACK); // Тло MNIST чорне
            g2.fillRect(0, 0, getSize().width, getSize().height);
            g2.setPaint(Color.WHITE); // Цифри білі
            // Важливо: товщина пензля. Для 280px полотна беремо товстий пензлик, 
            // щоб при зменшенні лінія залишалася видимою.
            g2.setStroke(new BasicStroke(22f, BasicStroke.CAP_ROUND, BasicStroke.JOIN_ROUND));
            repaint();
        }

        private void draw(int x1, int y1, int x2, int y2) {
            if (g2 != null) {
                g2.drawLine(x1, y1, x2, y2);
                repaint();
            }
        }

        // Перетворення малюнка у формат для нейромережі
        public double[] getMnistData() {
            // 1. Зменшуємо зображення до 28x28
            Image scaledImg = image.getScaledInstance(MNIST_SIZE, MNIST_SIZE, Image.SCALE_AREA_AVERAGING);
            BufferedImage smallImg = new BufferedImage(MNIST_SIZE, MNIST_SIZE, BufferedImage.TYPE_INT_RGB);
            Graphics g = smallImg.getGraphics();
            g.drawImage(scaledImg, 0, 0, null);
            g.dispose();

            // 2. Читаємо пікселі у масив double
            double[] input = new double[MNIST_SIZE * MNIST_SIZE];
            for (int y = 0; y < MNIST_SIZE; y++) {
                for (int x = 0; x < MNIST_SIZE; x++) {
                    int color = smallImg.getRGB(x, y);
                    // Отримуємо яскравість (Blue channel is enough for greyscale)
                    int brightness = (color & 0xff); 
                    input[y * MNIST_SIZE + x] = brightness / 255.0;
                }
            }
            return input;
        }
    }
}