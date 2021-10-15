package model.train;

import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.InvocationType;
import org.deeplearning4j.optimize.listeners.EvaluativeListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.lossfunctions.impl.LossMCXENT;

import java.io.File;
import java.io.IOException;

public class SignClassifierCustomCNN {
    private static double learningRate;
    private static int height, width, nChannel, nEpoch;
    private static int numClasses;
    private static MultiLayerNetwork model;
    private static DataSetIterator trainIter, testIter;

    public void setup(int height,
                 int width,
                 int nChannel,
                 int nEpoch,
                 int numClasses,
                 double learningRate,
                 DataSetIterator trainIter,
                 DataSetIterator testIter) {
        this.height = height;
        this.width = width;
        this.nChannel = nChannel;
        this.nEpoch = nEpoch;
        this.numClasses = numClasses;
        this.learningRate = learningRate;
        this.trainIter = trainIter;
        this.testIter = testIter;
    }

    public void trainModel() {
        UIServer server = UIServer.getInstance();
        StatsStorage storage = new InMemoryStatsStorage();
        server.attach(storage);

        model = new MultiLayerNetwork(getConfig());
        model.init();
        model.setListeners(
                new StatsListener(storage, 1),
                new ScoreIterationListener(100),
                new EvaluativeListener(trainIter, 1, InvocationType.EPOCH_END),
                new EvaluativeListener(testIter, 1, InvocationType.EPOCH_END)
        );
        model.fit(trainIter, nEpoch);
    }

    public void getEvaluation(boolean train) {
        System.out.println(train ? model.evaluate(trainIter).stats() : model.evaluate(testIter).stats());
    }

    public void saveModel(String path) throws IOException {
        ModelSerializer.writeModel(model, new File(path), true);
    }

    private MultiLayerConfiguration getConfig() {
        return new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(learningRate))
                .activation(Activation.RELU)
                .list()
                .layer(new ConvolutionLayer.Builder()
                        .nIn(nChannel)
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(32)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(64)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(128)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new BatchNormalization())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(256)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(256)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new ConvolutionLayer.Builder()
                        .kernelSize(3, 3)
                        .stride(1, 1)
                        .nOut(512)
                        .build())
                .layer(new SubsamplingLayer.Builder()
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .poolingType(SubsamplingLayer.PoolingType.MAX)
                        .build())
                .layer(new BatchNormalization())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(256)
                        .build())
                .layer(new DenseLayer.Builder()
                        .activation(Activation.RELU)
                        .nOut(60)
                        .build())
                .layer(new OutputLayer.Builder()
                        .activation(Activation.SOFTMAX)
                        .lossFunction(new LossMCXENT(Nd4j.create(new double[]{1, 1.16})))
                        .nOut(numClasses)
                        .build())
                .setInputType(InputType.convolutional(height, width, nChannel))
                .build();
    }
}
