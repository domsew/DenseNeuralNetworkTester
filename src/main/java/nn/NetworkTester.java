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
        MnistData mnist = new MnistData();
        DataSet trainData = mnist.getTrainData(10000, true);
        Network network = new Network(new int[] {784, 30, 10});
        network.setTestListener((int epoch, double accuracy, double loss) ->
                System.out.printf("%d\t%.4f\t%.4f\n", epoch, accuracy, loss));
        network.fit(trainData, 3, 10, 0.3, null);

//        INDArray x = network.predict(trainData.getFeatures(), true);
//        System.out.println(Arrays.toString(x.shape()));
//        System.out.println(Arrays.toString(x.argMax(1).shape()));

//        INDArray x = Nd4j.ones(10,784);
//        INDArray y = Nd4j.ones(10,50);
//        System.out.println(Arrays.toString(x.transpose().mmuli(y).shape()));
//        System.out.println(x.mmul(y));
    }

    static INDArray oneHot(INDArray input) {
        INDArray output = Nd4j.zeros(input.size(0), 10);
        for (int i = 0; i < input.size(0); i++) {
            output.put(i, input.getInt(i), 1);
        }
        return output;
    }
}
