package model.test;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.datavec.image.loader.ImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class SignTest {
    static int height = 224;
    static int width = 224;

    public static void main(String[] args) throws IOException {
        File CNNmodel = new File(System.getProperty("user.dir"), "generated-models/CNNmodel.zip");
        File VGG16model = new File(System.getProperty("user.dir"), "generated-models/VGG16model.zip");

        File testData = new ClassPathResource("sign_data/test/").getFile();
        ImageRecordReader recordReader = new ImageRecordReader(height, width, 1, new ParentPathLabelGenerator());
        recordReader.initialize(new FileSplit(testData));
        List<List<Writable>> fullData = new ArrayList<>();
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReader, 12, 1, 2);

        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(CNNmodel);

//        while (testIter.hasNext()) {
//            DataSet curr = testIter.next();
//
//            DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
//            scaler.transform(curr);
//
//            INDArray features = curr.getFeatures();
//            INDArray labels = curr.getLabels();
//
//            Evaluation evaluate = new Evaluation();
//            evaluate.eval(labels, model.output(features));
//            System.out.println(evaluate.stats());
//        }

        // Predict Images from Test Folder
        File testForge = new ClassPathResource("sign_data/test/forge/01_0101066.PNG").getFile();
        File testValid = new ClassPathResource("sign_data/test/valid/02_065.png").getFile();
        NativeImageLoader loader = new NativeImageLoader(height, width, 1);
        INDArray forgedSign = loader.asMatrix(testForge);
        INDArray validSign = loader.asMatrix(testValid);

        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(forgedSign);
        scaler.transform(validSign);

        INDArray output_1 = model.output(forgedSign);
        INDArray output_2 = model.output(validSign);

        System.out.println("========== FORGED SIGN ==========");
        System.out.println("Label: " + Nd4j.max(output_1, 1));
        System.out.println("Probailities: " + output_1.toString());
        System.out.println("Prediction: " + model.predict(forgedSign)[0]);

        System.out.println("========== VALID SIGN ==========");
        System.out.println("Label: " + Nd4j.max(output_2, 1));
        System.out.println("Probailities: " + output_2.toString());
        System.out.println("Prediction: " + model.predict(validSign)[0]);
    }
}
