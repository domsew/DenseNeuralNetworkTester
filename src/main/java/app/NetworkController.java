package app;

import javafx.application.Platform;
import javafx.beans.property.SimpleStringProperty;
import javafx.beans.property.StringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.fxml.FXMLLoader;
import javafx.fxml.Initializable;
import javafx.scene.Parent;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.XYChart;
import javafx.scene.control.CheckBox;
import javafx.scene.control.TableColumn;
import javafx.scene.control.TableView;
import javafx.scene.control.TextField;
import javafx.scene.control.cell.PropertyValueFactory;
import javafx.scene.control.cell.TextFieldTableCell;
import javafx.scene.text.Text;
import javafx.stage.Stage;
import javafx.util.StringConverter;
import nn.Mnist;
import nn.Network;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;

import java.io.IOException;
import java.net.URL;
import java.util.Arrays;
import java.util.ResourceBundle;

public class NetworkController implements Initializable {
    @FXML TableView layersTable;
    @FXML LineChart accuracyChart;
    @FXML LineChart lossChart;
    @FXML Text accuracyText;
    @FXML Text lossText;
    @FXML TextField epochsInput;
    @FXML TextField batchInput;
    @FXML TextField etaInput;
    @FXML TextField datasetSizeInput;
    @FXML CheckBox validateCheckBox;
    ObservableList<Layer> tableData = FXCollections.observableArrayList();
    private DataSet dataSet;
    int cnt;
    Thread thread;
    private Parent dataChooser;

    public void initialize(URL location, ResourceBundle resources) {
        tableInit();
    }

    @FXML void openDataset() throws IOException {
        layersTable.getScene().setRoot(getDataChooser());
    }

    @FXML void onStart() throws Exception {
        if (thread != null && thread.isAlive()) {
            return;
        }
        learnNetwork();
    }
    @FXML void onReset() {
        accuracyChart.getData().clear();
        lossChart.getData().clear();
    }
    @FXML void onAdd() {
        tableData.add(tableData.size() - 1, new Layer("FullyConnected", "30", "ReLU"));
    }
    @FXML void onRemove() {
        if (tableData.size() > 2) {
            tableData.remove(tableData.size() - 2);
        }
    }
    @FXML void onShowSamples() throws IOException {
        Stage stage = new Stage();
        stage.setTitle("Examples");
        stage.setScene(new Scene(new MnistSamplesPane(getDataSet())));
        stage.show();
    }

    private void learnNetwork() throws IOException {
        int epochs = Integer.parseInt(epochsInput.getText());
        int batch = Integer.parseInt(batchInput.getText());
        double eta = Double.parseDouble(etaInput.getText());
        int take = Integer.parseInt(datasetSizeInput.getText());
        boolean validate = validateCheckBox.isSelected();
        int[] sizes = new int[tableData.size()];
        int index = 0;
        for (Layer row : tableData) {
            sizes[index++] = Integer.parseInt(row.unitsProperty().getValue());
        }
        Network network = new Network(sizes);

        LineChart.Series testAccuracySeries = new LineChart.Series<>();
        testAccuracySeries.setName("Test " + cnt);
        LineChart.Series valAccuracySeries = new LineChart.Series<>();
        valAccuracySeries.setName("Validation " + cnt);
        LineChart.Series testLossSeries = new LineChart.Series<>();
        testLossSeries.setName("Test " + cnt);
        LineChart.Series valLossSeries = new LineChart.Series<>();
        valLossSeries.setName("Validation " + cnt);

        accuracyChart.getData().addAll(testAccuracySeries, valAccuracySeries);
        lossChart.getData().addAll(testLossSeries, valLossSeries);

        network.setTestListener((int epoch, double accuracy, double loss) -> {
            Platform.runLater(() -> testAccuracySeries.getData().add(new XYChart.Data<>(epoch, accuracy)));
            Platform.runLater(() -> testLossSeries.getData().add(new XYChart.Data<>(epoch, loss)));
        });
        network.setValidationListener((int epoch, double accuracy, double loss) -> {
            Platform.runLater(() -> valAccuracySeries.getData().add(new XYChart.Data<>(epoch, accuracy)));
            Platform.runLater(() -> valLossSeries.getData().add(new XYChart.Data<>(epoch, loss)));
        });
        DataSet ds = take > 0 ? getDataSet().sample(take) : getDataSet();
        DataSet trainData, valData;
        if (validate) {
            SplitTestAndTrain split = ds.splitTestAndTrain(0.9);
            trainData = split.getTrain();
            valData = split.getTest();
        } else {
            trainData = ds;
            valData = null;
        }

        thread = new Thread(() -> {
            double[] metrics = network.fit(trainData, epochs, batch, eta, valData);
            Platform.runLater(() -> accuracyText.setText( String.format("%.2f%%", metrics[0] * 100) ) );
            Platform.runLater(() -> lossText.setText( String.format("%.4f", metrics[1]) ) );
        });
        thread.start();
        cnt++;
    }

    public void setDataSet(DataSet ds, int inputSize) {
        tableData.get(0).units.setValue(Integer.toString(inputSize));
        dataSet = ds;
    }

    public DataSet getDataSet() throws IOException {
        return dataSet != null ?  dataSet : Mnist.load();
    }

    private Parent getDataChooser() throws IOException {
        if (dataChooser == null) {
            FXMLLoader loader = new FXMLLoader(getClass().getResource("/data_chooser.fxml"));
            dataChooser = loader.load();
            DataChooserController controller = loader.getController();
            controller.init(this, layersTable.getScene().getRoot());
        }
        return dataChooser;
    }

    private void tableInit() {
        tableData.addAll(
                new Layer("Input", "784", "None"),
                new Layer("FullyConnected", "30", "ReLU"),
                new Layer("FullyConnected", "10", "Softmax"));

        StringConverter<Object> sc = new StringConverter<Object>() {
            @Override
            public String toString(Object t) {
                return t == null ? null : t.toString();
            }

            @Override
            public Object fromString(String string) {
                return string;
            }
        };
        TableColumn typeCol = new TableColumn("Type");
        typeCol.setEditable(false);
        typeCol.setCellValueFactory(new PropertyValueFactory("type"));

        TableColumn unitsCol = new TableColumn("Neurons");
        unitsCol.setEditable(true);
        unitsCol.setCellValueFactory(new PropertyValueFactory("units"));
        unitsCol.setCellFactory(TextFieldTableCell.forTableColumn(sc));

        TableColumn activationCol = new TableColumn("Activation");
        activationCol.setEditable(false);
        activationCol.setCellValueFactory(new PropertyValueFactory("activation"));
        activationCol.setCellFactory(TextFieldTableCell.forTableColumn(sc));

        layersTable.getColumns().addAll(typeCol, unitsCol, activationCol);
        layersTable.setItems(tableData);
    }

    public void onClose(ActionEvent actionEvent) {
        Platform.exit();
    }

    public class Layer {
        private StringProperty type;
        private StringProperty units;
        private StringProperty activation;

        public Layer(String type, String units, String activation) {
            this.type = new SimpleStringProperty(type);
            this.units = new SimpleStringProperty(units);
            this.activation = new SimpleStringProperty(activation);
        }

        public StringProperty typeProperty() {
            return type;
        }

        public StringProperty unitsProperty() {
            return units;
        }

        public StringProperty activationProperty() {
            return activation;
        }
    }
}
