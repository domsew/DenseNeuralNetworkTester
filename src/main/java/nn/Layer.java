package nn;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ReluUniformInitScheme;

import java.util.Arrays;

public class Layer {
    INDArray weights;
    INDArray biases;
    INDArray inputs;
    INDArray logits;
    INDArray dW;
    INDArray dB;
    IActivation activation;

    public Layer(int units, int input_dim) {
        WeightInitScheme weightInit = new ReluUniformInitScheme('c', units);
        weights = weightInit.create(input_dim, units);
        biases = Nd4j.randn(1, units);
        activation = new ActivationReLU();
    }

    public INDArray call(INDArray input) {
        return call(input, false);
    }

    public INDArray call(INDArray input, boolean training) {
        INDArray z = input.mmul(weights).addiRowVector(biases);
        if (training) {
            inputs = input;
            logits = z;
        }
        return activation == null ? z : activation.getActivation(z, false);
    }

    public INDArray applyGrad(INDArray grad, double eta) {
        if (activation != null) {
            grad = activation.backprop(logits, grad).getFirst();
        }
        INDArray dW = inputs.transpose().mmul(grad);
        INDArray dB = grad.sum(0);

        double factor = eta / inputs.size(0);
        grad = weights.mmul(grad.transpose()).transpose();
        weights.subi(dW.muli(factor));
        biases.subi(dB.muli(factor));
        inputs = null;
        logits = null;

        return grad;
    }

    public void update(double eta, double miniBatchSize) {
        weights = weights.subi(dW.muli(eta/miniBatchSize));
        biases = biases.subi(dB.muli(eta/miniBatchSize));
        dW = null;
        dB = null;
        logits = null;
    }
}
