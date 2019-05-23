package app;

import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.Initializable;
import javafx.scene.Scene;
import javafx.scene.control.TextField;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.Stage;

import java.io.File;
import java.net.URL;
import java.util.List;
import java.util.ResourceBundle;

public class DataChooserController implements Initializable {
    @FXML TextField featuresSizeInput;
    @FXML TextField featuresHeaderInput;
    @FXML Text featuresFilesText;
    @FXML TextField labelsSizeInput;
    @FXML TextField labelsHeaderInput;
    @FXML Text labelsFilesText;
    private Stage stage;
    private Scene scene;
    private NetworkController parent;

    public void init(NetworkController parent, Scene scene, Stage stage) {
        this.parent = parent;
        this.scene = scene;
        this.stage = stage;
    }

    public void chooseFeaturesFile(ActionEvent actionEvent) {
        selectFile("features");
    }

    public void chooseLabelsFile(ActionEvent actionEvent) {
        selectFile("labels");
    }

    public void onLoad(ActionEvent actionEvent) {
        this.stage.setScene(this.scene);
    }

    @Override
    public void initialize(URL location, ResourceBundle resources) {
    }

    private List<File> selectFile(String title) {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setInitialDirectory(new File("."));
        fileChooser.setTitle(String.format("Select %s files", title));
        return fileChooser.showOpenMultipleDialog(this.stage);
    }
}
