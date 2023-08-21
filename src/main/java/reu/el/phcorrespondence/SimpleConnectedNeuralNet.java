package reu.el.phcorrespondence;

import lombok.Data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * SimpleConnectedNeuralNet - A3 simple multi-layer connected perceptron. The initial parameters of the network are
 * sampled from some probability distribution of type ParameterSampler. The activation function for each layer is
 * determined by the ActivationFunction functional interface. The number of layers, the number of neurons per layer,
 * and the number of classes are determined by the depth, width, and numClasses parameters respectively. The initial
 * input length is determined by the initialInputLength parameter. This neural net (NN) is implemented as a list of
 * SingleLayer objects. Each SingleLayer object contains the weights, biases, and activation function for a single
 * layer of the NN. The forward pass of the NN is implemented by the forwardPass method. The forwardPass method takes
 * in an array of doubles as input and returns an array of doubles as output. The length of the input array must be
 * equal to the initialInputLength parameter. The length of the output array is equal to the numClasses parameter.
 *
 * The purpose of this network is to be used as a simple classifier for the MNIST dataset or any abstract data set that
 * can be embedded in R^d x Z where d is the number of features and the labels correspond to the second coordinate Z
 * (and are thus discrete).
 *e
 * AUTHOR: Reuel Williams,
 *        Department of Mathematics,
 *        Princeton University
 */
@Data
public class SimpleConnectedNeuralNet {

    /**
     * SingleLayer - A class representing a single layer of a connected neural network. The number of neurons in the
     * layer is determined by the numNeurons parameter. The weights of each neuron are determined by the weights
     * parameter. The biases of each neuron are determined by the biases parameter. The activation function of the
     * layer is determined by the activationFunction parameter.
     */
    private class SingleLayer {
        /* fields for each layer of NN */
        private int numNeurons; /* number of neurons in layer */
        private double[][] weights; /* weights of each of the neurons. Weights of neuron i are in weights[i] */
        private double[] biases; /* biases of each of the neurons */
        private ActivationFunction activationFunction; /* activation function of each of the layer */

        /**
         * SingleLayer - Constructor for SingleLayer class
         *
         * @param numNeurons         : number of neurons in layer
         * @param weights            : weights of each of the neurons. Weights of neuron i are in weights[i]
         * @param biases             : biases of each of the neurons. Biases of neuron i are in biases[i]
         * @param activationFunction : activation function of each of the layer
         */
        private SingleLayer(int numNeurons, double[][] weights, double[] biases, ActivationFunction activationFunction) {

            /* Each neuron has the right number of weights */
            for (int i = 1; i < weights.length; i++) {
                if (weights[i].length != numNeurons)
                    throw new AssertionError("Number of weights for neuron" + i + "in layer must be equal to " +
                            "number of neurons");
            }
            /* Must have the right about of biases */
            assert biases.length == numNeurons : "Number of biases in layer must be equal to number of neurons";
            this.numNeurons = numNeurons;
            this.weights = weights;
            this.biases = biases;
            this.activationFunction = activationFunction;
        }

        /**
         * forwardPass - The forward pass of the neural network. Takes in an array of doubles as input and returns an
         * array of doubles as output. The length of the input array must be equal to the number of connections per
         * neuron. The length of the output array is equal to the number of neurons in the layer.
         *
         * @param inputs : array of doubles as input
         * @return : array of doubles as output
         */
        private double[] forwardPass(double[] inputs) {
            /* Must have the right number of inputs */
            assert inputs.length == weights[0].length : "length of input must correspond with the number of connections " +
                    "per neuron";

            double[] output = new double[numNeurons];
            for (int i = 0; i < this.numNeurons; i++) /* dot product of inputs and weights */
                for (int j = 0; j < inputs.length; j++) output[i] += weights[i][j] * inputs[j];
            /* adding biases and applying activation function */
            for (int i = 0; i < numNeurons; i++)
                output[i] = this.activationFunction.activationFunction(output[i] +
                        this.biases[i]);

            return output;
        }

    }

    /* fields for connected neural network */

    private List<SingleLayer> layers = new ArrayList<SingleLayer>(); /* layers in neural network */
    private int numParameters; /* number of parameters in neural network */
    private int depth; /* number of layers in neural network */
    private int numClasses; /* number of classes in neural network : possible discrete predictions */

    /**
     * SimpleConnectedNeuralNet - Constructor for SimpleConnectedNeuralNet class
     *
     * @param parameterSampler    : functional interface for sampling intial parameters
     * @param activationFunctions : list of activation functions for each layer
     * @param initialInputLength  : length of initial input
     */
    public SimpleConnectedNeuralNet(ParameterSampler parameterSampler, List<ActivationFunction> activationFunctions,
                                    List<Integer> hiddenLayerWidths, int initialInputLength, int numClasses) {
        /* Must have the right number of activation functions - one for each layer */
        assert activationFunctions.size() == hiddenLayerWidths.size() + 1 : "Number of activation functions must be " +
                "equal to number of layers";
        /* Must have some hidden layers*/
        assert hiddenLayerWidths.size() > 0 : "Number of layers must be greater than 0";
        /* Must have the right number of classes */
        assert numClasses > 0 : "Number of classes must be greater than 0";
        this.depth = hiddenLayerWidths.size() + 1; /* number of layers in neural network */
        this.numClasses = numClasses; /* number of classes in neural network */

        /* Initial parameters are provided as a vector in R^n where n is the number of parameters in the model
         * this is sampled using the ParameterSampler functional interface */
        int numParameters = 0;

        for (int i = 0; i < hiddenLayerWidths.size(); i++)
            /* number of parameters in each layer : weights + biases */
            numParameters += hiddenLayerWidths.get(i) * (i == 0 ? initialInputLength : hiddenLayerWidths.get(i - 1))
                    + hiddenLayerWidths.get(i);
        /* number of parameters in final layer : weights + biases */
        numParameters += numClasses * hiddenLayerWidths.get(hiddenLayerWidths.size() - 1) + numClasses;
        this.numParameters = numParameters;
        double[] initialParameters = parameterSampler.sampleParameters(numParameters);

        /* Initializing the hidden layers of the neural network */
        int used = 0; /* number of parameters used so far */
        for (int i = 0; i < depth - 1; i++) {
            int lengthOfPreviousInput = i == 0 ? initialInputLength : hiddenLayerWidths.get(i - 1);
            int width = hiddenLayerWidths.get(i);
            double[][] layerWeights = new double[width][lengthOfPreviousInput];
            double[] layerBiases = new double[width];
            /*  Initializing weights for layer i */
            for (int j = 0; j < width; j++) { /* initializing weights for neuron j in layer i */
                for (int k = 0; k < lengthOfPreviousInput; k++) {
                    layerWeights[j][k] = initialParameters[used + (j * lengthOfPreviousInput) + k];
                }
            }
            used += width * lengthOfPreviousInput;
            /* Initializing biases for layer i */
            for (int j = 0; j < width; j++) {
                layerBiases[j] = initialParameters[used + (i * width) + j];
            }
            used += width;
            /* Adding layer to list of layers */
            this.layers.add(new SingleLayer(width, layerWeights, layerBiases, activationFunctions.get(i)));
        }
        /* Initializing final layer */
        double[][] layerWeights = new double[numClasses][hiddenLayerWidths.get(hiddenLayerWidths.size() - 1)];
        double[] layerBiases = new double[numClasses];
        /*  Initializing weights for final layer */
        for (int j = 0; j < this.numClasses; j++) { /* initializing weights for neuron j in final layer*/
            for (int k = 0; k < numClasses; k++) {
                layerWeights[j][k] = initialParameters[used + (j * numClasses) + k];
            }
        }
        used += numClasses * hiddenLayerWidths.get(hiddenLayerWidths.size() - 1);
        /* Initializing biases for final layer */
        for (int j = 0; j < numClasses; j++) {
            layerBiases[j] = initialParameters[used + j];
        }
        /* Adding final layer to list of layers */
        this.layers.add(new SingleLayer(numClasses, layerWeights, layerBiases,
                activationFunctions.get(activationFunctions.size() - 1)));
    }

    /**
     * forwardPass - performs a forward pass through the neural network
     *
     * @param inputs : inputs to the neural network
     * @return
     */
    public double[] forwardPass(double[] inputs) {
        for (SingleLayer layer : this.layers) inputs = layer.forwardPass(inputs);
        return inputs;
    }

    /**
     * crossEntropyLoss - computes the cross entropy loss of the neural network
     *
     * @param inputs : inputs to the neural network
     * @return : cross entropy loss of the neural network
     */
    public double crossEntropyLoss(double[] inputs, int[] labels) {
        double[] predictions = this.forwardPass(inputs);
        double loss = 0;
        for (int i = 0; i < predictions.length; i++) loss += -Math.log(predictions[i]) * labels[i];
        return loss;
    }

    /**
     * SGD - performs stochastic gradient descent on the neural network
     *
     * @param inputs : inputs to the neural network
     */

    /**
     * getLayerWidth - returns the number of parameters in the neural network
     *
     * @param i : index of layer
     * @return : number of parameters in layer i
     */
    public int getLayerWidth(int i) {
        return this.layers.get(i).numNeurons;
    }

    /**
     * getLayerWeights - returns the weights of the ith layer
     *
     * @param i : index of layer
     * @return : number of parameters in layer i
     */
    public double[][] getLayerWeights(int i) {
        return this.layers.get(i).weights;
    }

    /**
     * getNormalizedLayerWeights - re
     */

    /**
     * getLayerBiases - returns the normalized weights of the ith layer
     *
     * @param i
     * @return
     */
    public double[][] getNormalizedLayerBiases(int i) {
        return Arrays.stream(this.layers.get(i).weights).map();
    }

    /**
     * getLayerActivationFunction - returns the number of parameters in the neural network
     *
     * @param i : index of layer
     * @return
     */
    public ActivationFunction getLayerActivationFunction(int i) {
        return this.layers.get(i).activationFunction;
    }

    /**
     * Private inner class the represents each neuron layer in the neural network by its layer and index in that layer.
     */
    public class Neuron {
        int layer; /* layer of neuron */
        int depth; /* index of neuron in layer */

        SimpleConnectedNeuralNet net;

        /**
         * Constructor for Neuron
         *
         * @param layer : layer of neuron
         * @param depth : index of neuron in layer
         */
        public Neuron(int layer, int depth, SimpleConnectedNeuralNet net) {
            this.layer = layer;
            this.depth = depth;
            this.net = net;
        }
    }

    /**
     * Returns the neurons in the neural network
     *
     * @return : the neurons in the neural network
     */
    public ArrayList<Neuron> getNeurons() {
        ArrayList<Neuron> neurons = new ArrayList<>();
        for (int i = 0; i < this.depth; i++) {
            for (int j = 0; j < this.getLayerWidth(i); j++) {
                neurons.add(new Neuron(i, j, this));
            }
        }
        return neurons;
    }
}

