package VGG_model;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.ColorConversionTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.core.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
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
import org.deeplearning4j.zoo.PretrainedType;
import org.deeplearning4j.zoo.ZooModel;
import org.deeplearning4j.zoo.model.VGG16;
import org.jetbrains.annotations.NotNull;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.primitives.Pair;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.awt.*;
import java.io.File;
import java.io.FilenameFilter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import static org.bytedeco.opencv.global.opencv_imgproc.CV_RGB2GRAY;

public class SignatureVerification {

    public static int height = 224;
    public static int width = 224;
    public static int channels = 3;
    public static int  seed = 123;
    public static double trainPerc = 0.8;
    public static int batchSize;
    public static int numClass = 2;
    public static int epoch = 3;
    public static Random rand =  new Random(seed);
    private static DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
    private static DataSetIterator trainIter, testIter;
    private static InputSplit trainData, testData;
    private static double lr = 0.001;

    private static Logger log = LoggerFactory.getLogger(SignatureVerification.class);

    private static ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();
    private static final String [] allowedExtensions = BaseImageLoader.ALLOWED_FORMATS;
    private static ImageTransform transform;


    public static void main(String[] args) throws Exception {

        //VGG16 setting
        ZooModel zooModel = VGG16.builder().build();
        ComputationGraph vgg16 = (ComputationGraph) zooModel.initPretrained(PretrainedType.IMAGENET);
        System.out.println(vgg16.summary());

        FineTuneConfiguration fnConf = new FineTuneConfiguration.Builder()
                .updater(new Adam(lr))
                .seed(seed)
                .build();

        ComputationGraph vgg16Transfer = new TransferLearning.GraphBuilder(vgg16)
                .fineTuneConfiguration(fnConf)
                .setFeatureExtractor("fc2")
                .removeVertexKeepConnections("predictions")
                .addLayer("predictions",
                        new OutputLayer.Builder()
                                .nIn(4096)
                                .nOut(numClass)
                                .weightInit(WeightInit.XAVIER)
                                .activation(Activation.SOFTMAX)
                                .lossFunction(LossFunctions.LossFunction.MCXENT)
                                .build(), "fc2")
                .build();

        System.out.println(vgg16.summary());

        setup(5, trainPerc, getTransform());

        trainIter = makeIterator(trainData);
        testIter = makeIterator(testData);

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        uiServer.attach(statsStorage);
        vgg16Transfer.setListeners(
                new StatsListener( statsStorage),
                new ScoreIterationListener(10)
        );

        vgg16Transfer.fit(trainIter, epoch);

        Evaluation evalTrain = vgg16Transfer.evaluate(trainIter);
        Evaluation evalTest = vgg16Transfer.evaluate(testIter);

        System.out.println(evalTrain.stats());
        System.out.println(evalTest.stats());

        File locationToSave = new File("generated-models","/signVerification.zip");
        log.info(locationToSave.toString());

        ModelSerializer.writeModel(vgg16Transfer, locationToSave, false);

    }

    private static RecordReaderDataSetIterator makeIterator(InputSplit split) throws Exception{
        ImageRecordReader imgRr = new ImageRecordReader(height, width, channels, labelMaker);
        imgRr.initialize(split, transform);

        RecordReaderDataSetIterator iter = new RecordReaderDataSetIterator(imgRr, batchSize, 1, numClass);
        iter.setPreProcessor(scaler);

        return iter;
    }

    private static ImageTransform getTransform(){
        ImageTransform rgb2gray = new ColorConversionTransform(CV_RGB2GRAY);
        List<Pair<ImageTransform, Double>> pipeline = Arrays.asList(
                new Pair<>(rgb2gray, 1.0)
        );
        return new PipelineImageTransform(pipeline, false);
    }

    private static void setup(int batchSizeArg, double trainRatio) throws IOException {
        batchSize = batchSizeArg;
        File imgPath = new ClassPathResource("sign_data/train").getFile();
        FileSplit imgFileSplit = new FileSplit(imgPath, NativeImageLoader.ALLOWED_FORMATS,rand);

        BalancedPathFilter filter = new BalancedPathFilter(rand, NativeImageLoader.ALLOWED_FORMATS, labelMaker);
        InputSplit[] imagesSplits = imgFileSplit.sample(filter, trainRatio, 1-trainRatio);

        trainData = imagesSplits[0];
        testData = imagesSplits[1];
    }

    public static void setup(int batchSizeArg, double trainRatio, ImageTransform imageTransform) throws IOException {
        transform = imageTransform;
        setup(batchSizeArg, trainRatio);
    }


}
