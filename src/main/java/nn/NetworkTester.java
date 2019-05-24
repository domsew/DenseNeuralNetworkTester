package nn;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.list.FloatNDArrayList;

import java.util.Arrays;

public class NetworkTester {
    public static void main(String[] args) throws Exception {
        DataSet dataSet = Mnist.load();
        DataSet trainData = dataSet.sample(10000);
        Network network = new Network(new int[] {784, 30, 10});
        network.setTestListener((int epoch, double accuracy, double loss) ->
                System.out.printf("%d\t%.4f\t%.4f\n", epoch, accuracy, loss));
        network.fit(trainData, 3, 10, 0.3, null);
//        INDArray x = network.predict(trainData.getFeatures(), true);
//        System.out.println(Arrays.toString(x.shape()));
//        System.out.println(Arrays.toString(x.argMax(1).shape()));
    }
}
