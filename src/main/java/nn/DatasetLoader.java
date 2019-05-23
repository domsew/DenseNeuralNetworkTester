package nn;

import org.apache.commons.io.FileUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.zip.GZIPInputStream;

public class DatasetLoader {

    public static DataSet load(Collection<File> featuresFiles, Collection<File> labelsFiles,
                        int featureLen, int featuresHeader, int labelsHeader) throws IOException {
        INDArray features = loadArray(featuresFiles, featureLen, featuresHeader);
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
        FileInputStream fis = new FileInputStream(file);
        fis.skip(header);
        return Nd4j.read(fis).reshape(-1, recordLen);
    }

    public static void unGzip(String file) throws IOException {
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

    public static void downloadAndUnGzip(String url, String dir, String ...filenames) throws IOException {
        for (String name : filenames) {
            if (new File(dir + name).exists()) { continue; }
            String gzName = name + ".gz";
            System.out.println("Downloading " + gzName + " from remote...");
            URL source = new URL(url + gzName);
            FileUtils.copyURLToFile(source, new File(dir + gzName));
            unGzip(dir + gzName);
        }
    }

    public static Collection<File> toFiles(String path, String ...filenames) {
        Collection<File> files = new ArrayList<>();
        for (String name : filenames) {
            files.add(new File(path, name));
        }
        return files;
    }
}
