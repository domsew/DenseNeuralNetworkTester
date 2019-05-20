package nn;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.shape.OneHot;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.ILossFunction;
import org.nd4j.linalg.lossfunctions.impl.*;

import java.util.List;

public class Network {
    private Layer[] layers;
    private NetworkEvent testHandler;
    private NetworkEvent validationHandler;
    ILossFunction loss;
    IActivation outActivation;
    int numClasses;

    public Network(int[] sizes) {
        layers = new Layer[sizes.length-1];
        layers[0] = new Layer(sizes[1], sizes[0]);
        for (int i = 1; i < sizes.length-1; i++) {
            layers[i] = new Layer(sizes[i+1], sizes[i]);
        }
        layers[layers.length - 1].activation = null;
        outActivation = new ActivationSoftmax();
        loss = new LossMCXENT();
        numClasses = sizes[sizes.length - 1];
    }

    public void setValidationListener(NetworkEvent handler) {
        validationHandler = handler;
    }
    public void setTestListener(NetworkEvent handler) {
        testHandler = handler;
    }

    public INDArray predict(INDArray input) {
        return predict(input, false);
    }

    public INDArray predict(INDArray input, boolean training) {
        for (Layer layer : layers) {
            input = layer.call(input, training);
        }
        return input;
//        return outActivation.getActivation(input, false);
    }

    public double[] fit(DataSet trainDataSet, int numEpoch, int batchSize, double eta, DataSet validationDataSet) {
        trainDataSet.shuffle();
        List<DataSet> batches = trainDataSet.batchBy(batchSize);
        double[] result = new double[2];

//        evaluate(trainDataSet, validationDataSet, 0);
        for (int i = 0; i < numEpoch; i++) {
            for (DataSet batch : batches) {
                INDArray pred = this.predict(batch.getFeatures(), true);
                INDArray grad = loss.computeGradient(oneHot(batch.getLabels()), pred, outActivation, null);
                backpropagation(grad, eta);
            }
            result = evaluate(trainDataSet, validationDataSet, i+1);
        }
        return result;
    }

    private void backpropagation(INDArray grad, double eta) {
        for (int l = layers.length - 1; l >= 0; l--) {
            grad = layers[l].applyGrad(grad, eta);
        }
    }

    private double[] evaluate(DataSet trainDataSet, DataSet validationDataSet, int i) {
        double[] result;
        if (trainDataSet.numExamples() > 10000) {
            result = calculateLossAndAccuracy((DataSet)trainDataSet.getRange(0, 10000));
        } else {
            result = calculateLossAndAccuracy(trainDataSet);
        }
        if (testHandler != null) {
            testHandler.call(i, result[0], result[1]);
        }
        if (validationDataSet != null && validationHandler != null) {
            result = calculateLossAndAccuracy(validationDataSet);
            validationHandler.call(i, result[0], result[1]);
        }

        return result;
    }

    private INDArray oneHot(INDArray input) {
        INDArray output = Nd4j.zeros(input.size(0), numClasses);
        for (int i = 0; i < input.size(0); i++) {
            output.put(i, input.getInt(i), 1);
        }
        return output;
    }

    private double[] calculateLossAndAccuracy(DataSet ds) {
        INDArray pred = predict(ds.getFeatures(), false);
        INDArray correct = pred.argMax(1).eqi(ds.getLabels()).sum();
        double score = loss.computeScore(oneHot(ds.getLabels()), pred, outActivation, null, true);

        return new double[]{ correct.getDouble(0) / ds.numExamples(), score };
    }
}
