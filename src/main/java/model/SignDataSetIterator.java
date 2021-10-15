package model;

import org.datavec.api.io.filters.BalancedPathFilter;
import org.datavec.api.io.filters.PathFilter;
import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.io.labels.PathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.image.loader.BaseImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.datavec.image.transform.FlipImageTransform;
import org.datavec.image.transform.ImageTransform;
import org.datavec.image.transform.PipelineImageTransform;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;

import java.io.File;
import java.io.IOException;
import java.util.Random;

public class SignDataSetIterator {

    private static int seed = 123;
    private static final Random randNumGen = new Random(seed);
    static String[] allowedFormats = BaseImageLoader.ALLOWED_FORMATS;
    static PathLabelGenerator labelGenerator = new ParentPathLabelGenerator();

    private static int height, width, nChannel, batchSize, numClasses;
    private static double trainFrac;
    private static ImageTransform transform;
    private static InputSplit trainData, testData;

    public void setup(int height, int width, int nChannel, int numClasses, double trainFrac, int batchSize, ImageTransform transform) throws IOException {
        this.height = height;
        this.width = width;
        this.nChannel = nChannel;
        this.numClasses = numClasses;
        this.trainFrac = trainFrac;
        this.batchSize = batchSize;
        this.transform = transform;

        File inputFile = new ClassPathResource("sign_data/train/").getFile();
        FileSplit fileSplit = new FileSplit(inputFile);

        PathFilter pathFilter = new BalancedPathFilter(randNumGen, allowedFormats, labelGenerator);
        InputSplit inputSplit[] = fileSplit.sample(pathFilter, trainFrac, 1-trainFrac);
        trainData = inputSplit[0];
        testData = inputSplit[1];
    }

    private DataSetIterator makeIterator(InputSplit split, boolean train) throws IOException {
        ImageRecordReader recordReader = new ImageRecordReader(height, width, nChannel, labelGenerator);
        if(train && transform != null) {
            recordReader.initialize(split, transform);
        } else {
            recordReader.initialize(split);
        }

        return new RecordReaderDataSetIterator(recordReader, batchSize, 1, numClasses);
    }

    public DataSetIterator getTrainIter() throws IOException {
        return makeIterator(trainData, true);
    }

    public DataSetIterator getTestIter() throws IOException {
        return makeIterator(testData, false);
    }
}
