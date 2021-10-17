package application;

import javafx.application.Application;
import javafx.event.ActionEvent;
import javafx.event.EventHandler;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Group;
import javafx.scene.Scene;
import javafx.scene.control.*;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.layout.BorderPane;
import javafx.scene.text.Text;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public class GUI extends Application implements EventHandler<ActionEvent> {

    private File imageFile;
    final FileChooser fileChooser = new FileChooser();
    final ImageView imageView = new ImageView();
    Alert alert = new Alert(Alert.AlertType.INFORMATION);
    private static MultiLayerNetwork model;
    private static final File modelFile = new File(System.getProperty("user.dir"), "generated-models/CNNmodel.zip");

    public static void main(String[] args) throws IOException {
        model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        launch(args);
    }

    @Override
    public void start(Stage primaryStage) throws Exception {
        MenuBar menuBar = new MenuBar();
        Menu fileMenu = new Menu("File");
        Menu helpMenu = new Menu("Help");

        MenuItem uploadItem = new MenuItem("Choose File");
        MenuItem exitItem = new MenuItem("Exit");
        MenuItem aboutItem = new MenuItem("About");

        menuBar.getMenus().addAll(fileMenu, helpMenu);
        fileMenu.getItems().addAll(uploadItem, exitItem);
        helpMenu.getItems().addAll(aboutItem);

        menuBar.setPadding(new Insets(0));

        uploadItem.setOnAction(e -> {
            File response = fileChooser.showOpenDialog(null);
            if (response != null) {
                imageFile = response.getAbsoluteFile();
                System.out.println(imageFile.getAbsolutePath());
                Image image = new Image(imageFile.toURI().toString());
                imageView.setImage(image);
                imageView.setFitHeight(280);
            }
        });

        exitItem.setOnAction(e -> System.exit(0));

        Label classLabel = new Label("Label: ");
        Label probabilityLabel = new Label("Probability: ");
        classLabel.setPadding(new Insets(5));
        probabilityLabel.setPadding(new Insets(5));


        Button button = new Button("Verify");
        button.setMaxSize(100, 37);
        button.setOnAction(this);

        BorderPane layout = new BorderPane();
        layout.setTop(menuBar);
        layout.setCenter(imageView);
        layout.setBottom(button);
        layout.setMargin(button, new Insets(0, 0, 10, 0));
        layout.setAlignment(button, Pos.CENTER);

        Scene scene = new Scene(layout, 640, 480);

        primaryStage.setScene(scene);
        primaryStage.setTitle("Signature Validator");
        primaryStage.show();
    }

    @Override
    public void handle(ActionEvent actionEvent) {
        if(imageFile != null) {
            NativeImageLoader loader = new NativeImageLoader(224, 224, 1);
            try {
                INDArray signImage = loader.asMatrix(imageFile);
                DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
                scaler.transform(signImage);
                INDArray output = model.output(signImage);
                int outClass = model.predict(signImage)[0];
                String label;
                if(outClass == 1)
                    label = "Valid";
                else
                    label = "Forged";
                System.out.println(Integer.toString(outClass));
                System.out.println(Nd4j.max(output, 1).toString());
                alert.setHeaderText(null);
                alert.setContentText(
                        "Label: " + label +
                        "\nProbability: " + Nd4j.max(output, 1).toString());
                alert.showAndWait();
            } catch (IOException e) {
                e.printStackTrace();
            }
        } else {
            alert.setHeaderText(null);
            alert.setContentText("* Please choose an image *");
            alert.showAndWait();
        }
    }

//    private static predict
}
