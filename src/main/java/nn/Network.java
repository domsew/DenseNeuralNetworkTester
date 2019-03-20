package nn;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationSoftmax;
import org.nd4j.linalg.api.ndarray.INDArray;
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

    public Network(int[] sizes) {
        layers = new Layer[sizes.length-1];
        layers[0] = new Layer(sizes[1], sizes[0]);
        for (int i = 1; i < sizes.length-1; i++) {
            layers[i] = new Layer(sizes[i+1], sizes[i]);
        }
        layers[layers.length - 1].activation = null;
        outActivation = new ActivationSoftmax();
        loss = new LossMCXENT();
    }

    public void setValidationListener(NetworkEvent handler) {
        validationHandler = handler;
    }
    public void setTestListener(NetworkEvent handler) {
        testHandler = handler;
    }

    public INDArray predict(INDArray x) {
        return predict(x, false);
    }
    public INDArray predict(INDArray x, boolean getLogits) {
        x = x.transpose();
        for (Layer layer : layers) {
            x = layer.predict(x);
        }
        if (getLogits) {
            return x;
        }
        return outActivation.getActivation(x, false);
    }

    public double[] fit(DataSet trainDataSet, int numEpoch, int batchSize, double eta, DataSet validationDataSet) {
        trainDataSet.shuffle();
        List<DataSet> batches = trainDataSet.batchBy(batchSize);
        double[] result = new double[2];

        evaluate(trainDataSet, validationDataSet, 0);
        for (int i = 0; i < numEpoch; i++) {
            for (DataSet batch : batches) {
                for (DataSet data : batch) {
                    backpropagation(data);
                }
                for (Layer layer : layers) {
                    layer.update(eta, batch.numExamples());
                }
            }
            result = evaluate(trainDataSet, validationDataSet, i+1);
        }
        return result;
    }

    private void backpropagation(DataSet data) {
        INDArray x = data.getFeatures().transpose();
        for (Layer layer : layers) {
            x = layer.activate(x);
        }
        INDArray gradient = loss.computeGradient(oneHot(data.getLabels()), x, outActivation, null);
        for (int l = layers.length - 1; l >= 0; l--) {
            gradient = layers[l].backward(gradient);
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

    private INDArray oneHot(INDArray y) {
        return Nd4j.zeros(10, 1).putScalar(y.toIntVector(),1);
    }

    private double[] calculateLossAndAccuracy(DataSet dataSet) {
        double correct = 0;
        double score = 0;
        for (DataSet data : dataSet) {
            INDArray z = predict(data.getFeatures(), true);
            INDArray y = data.getLabels();
            if (z.argMax().equals(y)) {
                correct++;
            }
            score += loss.computeScore(oneHot(y), z, outActivation, null, false);
        }
        return new double[]{ correct / dataSet.numExamples(), score / dataSet.numExamples() };
    }
}
