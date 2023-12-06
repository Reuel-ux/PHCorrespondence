package reu.el.phcorrespondence;

import edu.stanford.math.plex4.api.Plex4;
import edu.stanford.math.plex4.homology.barcodes.BarcodeCollection;
import edu.stanford.math.plex4.homology.interfaces.AbstractPersistenceAlgorithm;
import edu.stanford.math.plex4.metric.impl.EuclideanMetricSpace;
import edu.stanford.math.plex4.streams.impl.VietorisRipsStream;

import java.awt.*;
import java.util.List;

public class runNN {

    private int prediction(double[] input) {
        int maxIndex = 0;
        double max = input[0];
        for (int i = 1; i < input.length; i++) {
            if (input[i] > max) {
                max = input[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public BarcodeCollection netPersistence(double maxFiltrationValue, int maxDimension, int torsion, SimpleConnectedNeuralNet nn){
        SimpleConnectedNeuralNet.Neuron[] neurons = nn.getNeurons();
        InfluenceMetricSpace nnMetricSpace = new InfluenceMetricSpace(neurons);
        /* creates filtered VR complex that can detect cycles up to dimension <maxDimension> */
        VietorisRipsStream stream = new VietorisRipsStream(nnMetricSpace, maxFiltrationValue, maxDimension + 1);
        stream.finalizeStream();

        AbstractPersistenceAlgorithm persistence = Plex4.getModularSimplicialAlgorithm(maxDimension + 1, torsion);
        return persistence.computeIntervals(stream);
    }

    public static void main(String[] args) {
        int initialInputLength = 784; /* length of MNIST input*/
        int numClasses = 2; /* number of classes - numbers 0-9 */
        ParameterSampler sampler = new PointSampler.randomSampler(1).getSampler(); /* parameter sampler */
        ActivationFunction activation = Activation.ReLu; /* activation function */
        List<Integer> hiddenLayerWidths = List.of(2, 2); /* hidden layer widths */
        List<ActivationFunction> hiddenActivations = List.of(activation, activation); /* hidden layer activations */

        /* creating a new simply connected neural network */
        SimpleConnectedNeuralNet nn = new SimpleConnectedNeuralNet(sampler,
                hiddenActivations, hiddenLayerWidths, initialInputLength, numClasses);

        /* embedding MNIST data*/
        String pathToPhotos = "src/main/resources/MNIST-TEST/0";
        String trainingImages = "src/main/resources/MNIST-TRAIN/";
        /* creating embeddings to train network */
        for (int i = 0; i <= 1; i++){
            double[] losses;
            String path = trainingImages + i;
            Embeddings MNISTEmbeddings = new Embeddings(path);
            int[] labels = MNISTEmbeddings.getLabels();

            /* one hot encoding */
            double[][] correctOutputs = new double[labels.length][numClasses];
            for (int j = 0; j < labels.length; j++) {
                correctOutputs[j][labels[j]] = 1;
            }
            losses = nn.train(MNISTEmbeddings.getEmbeddings(), correctOutputs, 0.5);
            /* seeing losses */
            for (int j = 0; j < losses.length; j++) System.out.println(losses[j]);
        }


        Embeddings MNISTEmbeddings = new Embeddings(pathToPhotos);

        runNN test = new runNN();
        int prediction = test.prediction(nn.forwardPass(MNISTEmbeddings.getEmbeddings()[0]));
        System.out.println(prediction);

       /* printing our barcode of Neural Net */
//        BarcodeCollection barcode = test.netPersistence(100, 10, 2, nn);
//System.out.println(barcode);

    }

}
