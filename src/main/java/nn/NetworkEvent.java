package nn;

public interface NetworkEvent {
    void call(int epoch, double accuracy, double loss);
}
