package model.train;

import model.SignDataSetIterator;
import org.datavec.image.transform.*;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.ResNet50;
import org.deeplearning4j.zoo.model.VGG16;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

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
    private static ImageTransform transform;

    public static void main(String[] args) throws IOException {

        transform = new PipelineImageTransform.Builder()
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

//        CNN();
        TransferLearning();
    }

    private static void CNN() throws IOException {
        DataNormalization imagePreProcessor = new ImagePreProcessingScaler();
        imagePreProcessor.fit(trainIter);
        trainIter.setPreProcessor(imagePreProcessor);
        testIter.setPreProcessor(imagePreProcessor);

        SignClassifierCustomCNN CNNclassifier = new SignClassifierCustomCNN();
        CNNclassifier.setup(
                height,
                width,
                nChannel,
                nEpoch,
                numClasses,
                learningRate,
                trainIter,
                testIter
        );
        CNNclassifier.trainModel();
        System.out.println("========== TRAINING EVALUATION ==========");
        CNNclassifier.getEvaluation(true);
        System.out.println("========== TESTING EVALUATION ==========");
        CNNclassifier.getEvaluation(false);
        CNNclassifier.saveModel("generated-models/CNNmodel.zip");
    }

    private static void TransferLearning() throws IOException {
        ImageTransform transformTL = new PipelineImageTransform.Builder()
                .addImageTransform(new FlipImageTransform(1), 0.4)
                .addImageTransform(new RotateImageTransform(15), 0.3)
                .addImageTransform(new RotateImageTransform(25), 0.3)
//                .addImageTransform(new ScaleImageTransform((float) 0.7), 0.3)
                .addImageTransform(new ShowImageTransform("Augment", 500), 1.0)
                .build();

        SignDataSetIterator dataSetIteratorTL = new SignDataSetIterator();
        dataSetIteratorTL.setup(height, width, 3, numClasses, trainFrac, batchSize, transformTL);

        DataSetIterator trainIterTL = dataSetIteratorTL.getTrainIter();
        DataSetIterator testIterTL = dataSetIteratorTL.getTestIter();

        // Normalization
        DataNormalization preProcessor = new VGG16ImagePreProcessor();
        preProcessor.fit(trainIterTL);
        trainIterTL.setPreProcessor(preProcessor);
        testIterTL.setPreProcessor(preProcessor);

        DataNormalization vggPreProcessor = new ImagePreProcessingScaler();
        ZooModel zooModel = VGG16.builder().build(); // Can change to other pre trained model
        ComputationGraph pretrainModel = (ComputationGraph) zooModel.initPretrained();

        SignClassifierTransferLearning VGG16classifier = new SignClassifierTransferLearning();
        VGG16classifier.setup(
                height,
                width,
                nChannel,
                nEpoch,
                numClasses,
                learningRate,
                trainIterTL,
                testIterTL,
                pretrainModel
        );
        VGG16classifier.trainModel();
        System.out.println("========== TRAINING EVALUATION ==========");
        VGG16classifier.getEvaluation(true);
        System.out.println("========== TESTING EVALUATION ==========");
        VGG16classifier.getEvaluation(false);
        VGG16classifier.saveModel("generated-models/VGG16model.zip");
    }
}
