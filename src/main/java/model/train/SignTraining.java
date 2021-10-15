package model.train;

import model.SignDataSetIterator;
import org.datavec.image.transform.*;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.IOException;
import java.util.Random;

public class SignTraining {
    static double learningRate = 0.001;
    static int nEpoch = 10;
    static double trainFrac = 0.8;

    static int height = 224;
    static int width = 224;
    static int nChannel = 1;
    private static int batchSize = 8;
    private static int numClasses = 2;
    private static DataSetIterator trainIter, testIter;

    public static void main(String[] args) throws IOException {

        ImageTransform transform = new PipelineImageTransform.Builder()
                .addImageTransform(new FlipImageTransform(1), 0.4)
                .addImageTransform(new RotateImageTransform(15), 0.3)
                .addImageTransform(new RotateImageTransform(25), 0.3)
                .addImageTransform(new ScaleImageTransform((float) 0.7), 0.3)
                .addImageTransform(new ShowImageTransform("Augment", 500), 1.0)
                .build();

        SignDataSetIterator dataSetIterator = new SignDataSetIterator();
        dataSetIterator.setup(height, width, nChannel, numClasses, trainFrac, batchSize, transform);

        trainIter = dataSetIterator.getTrainIter();
        testIter = dataSetIterator.getTestIter();
        CNN();
    }

    private static void CNN() throws IOException {
        SignClassifierCustomCNN CNNclassifier = new SignClassifierCustomCNN();
        CNNclassifier.setup(height, width, nChannel, nEpoch, numClasses, learningRate, trainIter, testIter);
        CNNclassifier.trainModel();
        System.out.println("========== TRAINING EVALUATION ==========");
        CNNclassifier.getEvaluation(true);
        System.out.println("========== TESTING EVALUATION ==========");
        CNNclassifier.getEvaluation(false);
        CNNclassifier.saveModel("generated-models/CNNmodel.zip");
    }
}
