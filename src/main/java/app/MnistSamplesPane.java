package app;

import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.PixelFormat;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.Pane;
import nn.MnistData;
import org.nd4j.linalg.dataset.DataSet;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.Iterator;

public class MnistSamplesPane extends Pane {

    private Canvas canvas;
    private MnistData mnist;
    private GraphicsContext gc;
    private int height = 28;
    private int width = 28;

    public MnistSamplesPane(MnistData mnist) throws IOException {
        this.mnist = mnist;
        canvas = new Canvas();
        canvas.setHeight(305);
        canvas.setWidth(305);
        gc = canvas.getGraphicsContext2D();
        setStyle("-fx-background-color: black");
        getChildren().add(canvas);
        drawSamples();
    }

    private void drawSamples() throws IOException {
        double k = Math.random()*59600;
        Iterator<DataSet> data = mnist.getTrainData((int)k, 25, false).iterator();

        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                drawImage(data.next(), x * width * 2 + 5, y * height * 2 + 5);
            }
        }
    }

    private void drawImage(DataSet data, int x, int y) {
        float[] imgBuf = data.getFeatures().toFloatVector();
        int numChannels = 3;

        WritableImage writableImage = new WritableImage(width, height);
        PixelWriter pixelWriter = writableImage.getPixelWriter();
        PixelFormat<ByteBuffer> pixelFormat = PixelFormat.getByteRgbInstance();
        byte[] buffer = new byte[height * width * numChannels];
        // Arrays.fill(buffer, (byte)127);

        for (int r = 0; r < height; r++) {
            for (int c = 0; c < width; c++) {
                int i = r * height + c;
                byte value = (byte)(imgBuf[i]);
                buffer[i * numChannels + 0] = value;
                buffer[i * numChannels + 1] = value;
                buffer[i * numChannels + 2] = value;
            }
        }
        pixelWriter.setPixels(0, 0, width, height, pixelFormat, buffer, 0, width * numChannels);

        gc.drawImage(writableImage, x, y, width*2, height*2);
    }
}
