package app;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.control.TextField;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import nn.DatasetLoader;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.ResourceBundle;

public class DataChooserController implements Initializable {
    @FXML TextField featuresSizeInput;
    @FXML TextField featuresHeaderInput;
    @FXML Text featuresFilesText;
    @FXML TextField labelsHeaderInput;
    @FXML Text labelsFilesText;
    private NetworkController parentController;
    private Parent parentRoot;
    private Collection<File> featuresFiles;
    private Collection<File> labelsFiles;

    public void init(NetworkController parentController, Parent parentRoot) {
        this.parentController = parentController;
        this.parentRoot = parentRoot;
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {}

    public void chooseFeaturesFile(ActionEvent actionEvent) {
        featuresFiles = selectFile("features");
        featuresFilesText.setText(filesToString(featuresFiles));
    }

    public void chooseLabelsFile(ActionEvent actionEvent) {
        labelsFiles = selectFile("labels");
        labelsFilesText.setText(filesToString(labelsFiles));
    }

    public void onLoad(ActionEvent actionEvent) throws IOException {
        if (featuresFiles == null || labelsFiles == null) { return; }
        int featuresSize = Integer.parseInt(featuresSizeInput.getText());
        int featuresHeader = parseOrDefault(featuresHeaderInput.getText());
        int labelsHeader = parseOrDefault(labelsHeaderInput.getText());
        DataSet dataSet = DatasetLoader.load(featuresFiles, labelsFiles, featuresSize, featuresHeader, labelsHeader);
        parentController.setDataSet(dataSet, featuresSize);
        featuresFilesText.getScene().setRoot(parentRoot);
    }

    public void onCancel(ActionEvent actionEvent) {
        featuresFilesText.getScene().setRoot(parentRoot);
    }

    private static String filesToString(Collection<File> files) {
        if (files == null)
            return "";
        List<String> names = new ArrayList<>();
        for (File file : files) {
            names.add(file.getName());
        }
        return String.join(", ", names);
    }

    private static int parseOrDefault(String value) {
        int ret;
        try {
            ret = Integer.parseInt(value, 10);
        } catch(NumberFormatException ex) {
            ret = 0;
        }
        return ret;
    }

    private List<File> selectFile(String title) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setInitialDirectory(new File("."));
        fileChooser.setTitle(String.format("Select %s files", title));
        return fileChooser.showOpenMultipleDialog(null);
    }
}
