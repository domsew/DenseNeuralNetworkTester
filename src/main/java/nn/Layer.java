package nn;

import org.nd4j.linalg.activations.IActivation;
import org.nd4j.linalg.activations.impl.ActivationReLU;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.weightinit.WeightInitScheme;
import org.nd4j.weightinit.impl.ReluUniformInitScheme;

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
        weights = weightInit.create(units, input_dim);
        biases = Nd4j.randn(units, 1);
        activation = new ActivationReLU();
    }

    public INDArray predict(INDArray x) {
        INDArray z = weights.mmul(x).addi(biases);
        if (activation == null) {
            return z;
        }
        return activation.getActivation(z, false);
    }

    public INDArray activate(INDArray x) {
        inputs = x;
        logits = weights.mmul(inputs).addi(biases);
        if (activation == null) {
            return logits;
        }
        return activation.getActivation(logits, false);
    }

    public INDArray backward(INDArray gradient) {
        if (activation != null) {
            gradient = activation.backprop(logits, gradient).getFirst();
        }
        if (dW == null) {
            dW = gradient.mmul(inputs.transpose());
            dB = gradient;
        } else {
            dW.addi(gradient.mmul(inputs.transpose()));
            dB.addi(gradient);
        }
        return weights.transpose().mmul(gradient);
    }

    public void update(double eta, double miniBatchSize) {
        weights = weights.subi(dW.muli(eta/miniBatchSize));
        biases = biases.subi(dB.muli(eta/miniBatchSize));
        dW = null;
        dB = null;
        logits = null;
    }
}
