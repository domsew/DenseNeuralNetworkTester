/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package nn;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.zip.GZIPInputStream;

import org.apache.commons.compress.compressors.gzip.GzipUtils;
import org.apache.commons.io.FileUtils;
import org.nd4j.compression.impl.Gzip;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.dataset.DataSet;

/**
 *
 * @author dominik.seweryn
 */
public class MnistData {
    private final String baseUrl = "http://yann.lecun.com/exdb/mnist/";
    private final String basePath = "temp/";
    private final String trainImageFile = "train-images-idx3-ubyte.gz";
    private final String trainLabelFile = "train-labels-idx1-ubyte.gz";
    private final String testImageFile = "t10k-images-idx3-ubyte.gz";
    private final String testLabelFile = "t10k-labels-idx1-ubyte.gz";

    public MnistData() throws IOException {
        download();
    }
    
    public DataSet getTrainData(boolean normalized) throws Exception {
        return getTrainData(null, null, normalized);
    }

    public DataSet getTrainData(Integer take, boolean normalized) throws Exception {
        return getTrainData(null, take, normalized);
    }

    public DataSet getTrainData(Integer skip, Integer take, boolean normalized) throws Exception {
        return loadDataSet(trainImageFile, trainLabelFile, skip, take, normalized);
    }
    
    public DataSet getTestData(boolean normalized) throws Exception {
        return loadDataSet(testImageFile, testLabelFile, null, null, normalized);
    }
    
    private DataSet loadDataSet(String imagesFile, String labelsFile, Integer skip, Integer take, boolean normalized) throws Exception {
        if (take != null && take <= 0) {
            throw new Exception("Take cant be <= 0");
        }
        INDArray images = readImages(basePath + imagesFile, skip, take);
        INDArray labels = readLabels(basePath + labelsFile, skip, take);
        if (normalized) {
            images.divi(255.0);
        }
        return new DataSet(images, labels);
    }
    
    private void download() throws IOException {
        String[] files = {trainImageFile, trainLabelFile, testImageFile, testLabelFile};
        
        for (String file : files) {
            String filePath = basePath + file;
            if (new File(filePath).exists()) {
                continue;
            }
            URL source = new URL(baseUrl + file);
            System.out.println("Downloading " + file + " from remote...");
            FileUtils.copyURLToFile(source, new File(filePath));
        }
    }
    
    private INDArray readLabels(String fileName, Integer skip, Integer take) throws IOException {
        InputStream gzip = new GZIPInputStream(new FileInputStream(fileName));
        
        byte[] intBuf = new byte[4];
        gzip.read(intBuf);
        assert ByteBuffer.wrap(intBuf).getInt() == 2049;
        
        gzip.read(intBuf);
        int count = ByteBuffer.wrap(intBuf).getInt();
        if (skip != null) {
            gzip.skip(skip);
        }
        if (take != null) {
            count = take;
        }
        assert count == gzip.available();
        
        float[] buffer = new float[count];
        for (int i = 0; i < count; i++) {
            buffer[i] = gzip.read();
        }
        return Nd4j.create(buffer, new int[]{count, 1});
    }
    
    private INDArray readImages(String fileName, Integer skip, Integer take) throws IOException {
        InputStream gzip = new GZIPInputStream(new FileInputStream(fileName));
        
        byte[] intBuf = new byte[4];
        gzip.read(intBuf);
        assert ByteBuffer.wrap(intBuf).getInt() == 2051;
        
        gzip.read(intBuf);
        int count = ByteBuffer.wrap(intBuf).getInt();
        gzip.read(intBuf);
        int height = ByteBuffer.wrap(intBuf).getInt();
        gzip.read(intBuf);
        int width = ByteBuffer.wrap(intBuf).getInt();
        int numPixels = height * width;
        if (skip != null) {
            gzip.skip(skip * numPixels);
        }
        if (take != null) {
            count = take;
        }
        assert count * numPixels == gzip.available();
        
        float[] buffer = new float[count * numPixels];
        for (int r = 0; r < count * numPixels; r++) {
            buffer[r] = gzip.read();
        }
        return Nd4j.create(buffer, new int[]{count, numPixels});
    }
}
