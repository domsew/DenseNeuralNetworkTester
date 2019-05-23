package nn;

import org.nd4j.linalg.dataset.DataSet;

import java.io.*;
import java.util.Collection;

public class Mnist {
    private static final String baseUrl = "http://yann.lecun.com/exdb/mnist/";
    private static final String basePath = "temp/";
    private static final String trainImageFile = "train-images-idx3-ubyte";
    private static final String trainLabelFile = "train-labels-idx1-ubyte";
    private static final String testImageFile = "t10k-images-idx3-ubyte";
    private static final String testLabelFile = "t10k-labels-idx1-ubyte";

    public static DataSet load() throws IOException {
        Collection<File> features = DatasetLoader.toFiles(basePath, trainImageFile, testImageFile);
        Collection<File> labels = DatasetLoader.toFiles(basePath, trainLabelFile, testLabelFile);
        return DatasetLoader.load(features, labels, 784, 16, 8);
    }

    public static void get() throws IOException {
        String[] files = {trainImageFile, trainLabelFile, testImageFile, testLabelFile};
        DatasetLoader.downloadAndUnGzip(baseUrl, basePath, files);
    }

    public static int[] isMnist(File file) throws IOException {
        DataInputStream dis = new DataInputStream(new FileInputStream(file));
        int shape[] = null;
        // int count, height, width;
        int magic = dis.readInt();
        if (magic == 2051) {
            int count = dis.readInt();
            int height = dis.readInt();
            int width = dis.readInt();
            shape = new int[]{ count, height, width };
        } else if (magic == 2049) {
            int count = dis.readInt();
            shape = new int[]{ count };
        }
        dis.close();
        return shape;
    }
}
