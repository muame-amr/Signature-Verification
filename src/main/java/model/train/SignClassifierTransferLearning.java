package model.train;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.transferlearning.FineTuneConfiguration;
import org.deeplearning4j.nn.transferlearning.TransferLearning;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossNegativeLogLikelihood;

import java.io.File;
import java.io.IOException;

public class SignClassifierTransferLearning {
    private static double learningRate;
    private static int height, width, nChannel, nEpoch;
    private static int numClasses;
    private static ComputationGraph model, preTrain;
    private static DataSetIterator trainIter, testIter;
    private static String feLayer = "fc2";
    private static String outputLayer = "sign_output";

    public void setup(int height,
                      int width,
                      int nChannel,
                      int nEpoch,
                      int numClasses,
                      double learningRate,
                      DataSetIterator trainIter,
                      DataSetIterator testIter,
                      ComputationGraph preTrain) {
        this.height = height;
        this.width = width;
        this.nChannel = nChannel;
        this.nEpoch = nEpoch;
        this.numClasses = numClasses;
        this.learningRate = learningRate;
        this.trainIter = trainIter;
        this.testIter = testIter;
        this.preTrain = preTrain;
    }

    public void trainModel() {

        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);

        model = getTransferModel();
        System.out.println(model.summary());
        model.setListeners(
                new ScoreIterationListener(10),
                new StatsListener(storage, 10),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );
        model.fit(trainIter, nEpoch);
    }

    private FineTuneConfiguration getFineTuneConfig() {
        return new FineTuneConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .seed(123)
                .build();
    }

    private ComputationGraph getTransferModel() {
        return new TransferLearning.GraphBuilder(preTrain)
                .fineTuneConfiguration(getFineTuneConfig())
//                .nInReplace("block1_conv1", nChannel, WeightInit.XAVIER)
                .setFeatureExtractor(feLayer)
                .removeVertexAndConnections("predictions")
                .addLayer(outputLayer,
                        new OutputLayer.Builder()
                                .nIn(4096).nOut(numClasses)
                                .lossFunction(new LossNegativeLogLikelihood(Nd4j.create(new double[]{1, 0.8591})))
                                .activation(Activation.SOFTMAX)
                                .build(), feLayer)
                .setOutputs(outputLayer)
                .build();
    }

    public void getEvaluation(boolean train) {
        System.out.println(train ? model.evaluate(trainIter).stats() : model.evaluate(testIter).stats());
    }

    public void saveModel(String path) throws IOException {
        ModelSerializer.writeModel(model, new File(path), true);
    }
}
