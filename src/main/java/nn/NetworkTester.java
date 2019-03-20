package nn;

import org.nd4j.linalg.dataset.DataSet;
import java.io.IOException;

public class NetworkTester {
    public static void main(String[] args) throws IOException {
        MnistData mnist = new MnistData();
        DataSet trainData = mnist.getTrainData(1000, true);
        Network network = new Network(new int[] {784, 50, 10});
        network.setTestListener((int epoch, double accuracy, double loss) -> {
            System.out.printf("%d\t%.4f\t%.4f\n", epoch, accuracy, loss);
        });
        network.fit(trainData, 3, 10, 0.1, null);
    }
}
