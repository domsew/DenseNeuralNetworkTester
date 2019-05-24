package nn;

import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.URL;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.zip.GZIPInputStream;

public class DatasetLoader {

    public static DataSet load(Collection<File> featuresFiles, Collection<File> labelsFiles,
                        int featureLen, int featuresHeader, int labelsHeader) throws IOException {
        INDArray features = loadArray(featuresFiles, featureLen, featuresHeader).divi(255);
        INDArray labels = loadArray(labelsFiles, 1, labelsHeader);
        return new DataSet(features, labels);
    }

    private static INDArray loadArray(Collection<File> files, int recordLen, int header) throws IOException {
        List<INDArray> arrays = new ArrayList<>();
        for (File file : files) {
            arrays.add(loadArray(file, recordLen, header));
        }
        return Nd4j.vstack(arrays);
    }

    private static INDArray loadArray(File file, int recordLen, int header) throws IOException {
        InputStream is = new FileInputStream(file);
        is.skip(header);
        byte[] buffer = IOUtils.toByteArray(is);
        is.close();
        float[] array = new float[buffer.length];
        for (int i = 0; i < buffer.length; i++)
            array[i] = 0xFF & buffer[i];
        return Nd4j.create(array, new int[]{ buffer.length / recordLen, recordLen });
    }

    private static void unGzip(String file) throws IOException {
        File gzFile = new File(file + ".gz");
        GZIPInputStream gis = new GZIPInputStream(new FileInputStream(gzFile));
        FileOutputStream fos = new FileOutputStream(file);
        byte[] buffer = new byte[8192];
        int len;
        while((len = gis.read(buffer)) != -1){
            fos.write(buffer, 0, len);
        }
        gis.close();
        fos.close();
        gzFile.delete();
    }

    static void downloadAndUnGzip(String url, String dir, String ...filenames) throws IOException {
        for (String name : filenames) {
            if (new File(dir + name).exists()) { continue; }
            String gzName = name + ".gz";
            System.out.println("Downloading " + gzName + " from remote...");
            URL source = new URL(url + gzName);
            FileUtils.copyURLToFile(source, new File(dir + gzName));
            System.out.println("Extracting " + gzName + "...");
            unGzip(dir + name);
            System.out.println("Done " + name);
        }
    }

    static Collection<File> toFiles(String path, String ...filenames) {
        Collection<File> files = new ArrayList<>();
        for (String name : filenames) {
            files.add(new File(path, name));
        }
        return files;
    }
}
