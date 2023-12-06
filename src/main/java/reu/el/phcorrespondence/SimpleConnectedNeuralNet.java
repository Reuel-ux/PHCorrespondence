package reu.el.phcorrespondence;

import lombok.Data;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * SimpleConnectedNeuralNet - A simple multi-layer connected perceptron. The initial parameters of the network are
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
 *
 * AUTHOR: Reuel Williams,
 *        Department of Mathematics,
 *        Princeton University
 */
@Data
public class SimpleConnectedNeuralNet {

    /**
     * SingleLayer - A class representing a single layer of a connected neural network. The number of neurons in the
     * layer is determined by the numNeurons parameter. The weights of each neuron are determined by the 'weights'
     * parameter. The biases of each neuron are determined by the 'biases' parameter. The activation function of the
     * layer is determined by the 'activationFunction' parameter.
     */
    @Data
    public static class SingleLayer {
        /* fields for each layer of NN */
        private int numNeurons; /* number of neurons in layer */
        private double[][] weights; /* weights of each of the neurons. Weights of neuron i are in weights[i] */
        private double[] biases; /* biases of each of the neurons */
        private ActivationFunction activationFunction; /* activation function of each of the layer */

        /**
         * SingleLayer - Constructor for SingleLayer class
         *
         * @param numNeurons         : number of neurons in layer
         * @param weights            : weights of each of the neurons. Weights of neuron i are in weights[i][]
         * @param biases             : biases of each of the neurons. Biases of neuron i are in biases[i]
         * @param activationFunction : activation function of each of the layer
         */

        private SingleLayer(int numNeurons, double[][] weights, double[] biases, ActivationFunction activationFunction) {
            /* Must be weights for each neuron */
            assert weights.length == numNeurons : "Number of weights in layer must be equal to number of neurons";

            /* each neuron must have the same number of weights */
            for (double[] weight : weights) {
                assert weight.length == weights[0].length : "Each neuron must have the same number of weights";
            }
            /* activation function cannot be null */
            assert activationFunction != null : "Activation function cannot be null";
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
            for (int i = 0; i < numNeurons; i++)  output[i] = this.activationFunction.activationFunction(output[i] +
                        this.biases[i]);

            return output;
        }

    }

    /* fields for connected neural network */

    private List<SingleLayer> layers = new ArrayList<>(); /* layers in neural network */
    private int numParameters; /* number of parameters in neural network */
    private int depth; /* number of layers in neural network */
    private int numClasses; /* number of classes in neural network : possible discrete predictions */
    private double[][] activations; /* activations of each layer when forward pass is made */

    /**
     * SimpleConnectedNeuralNet - Constructor for SimpleConnectedNeuralNet class
     *
     * @param parameterSampler    : functional interface for sampling initial parameters
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
        this.numClasses = numClasses; /* number of classes that the neural network can classify inputs for */
        this.activations = new double[depth + 1][]; /* activations of each layer when forward pass is made */

        /* Initial parameters are provided as a vector in R^n where n is the number of parameters in the model
         * this is sampled using the ParameterSampler functional interface */
        int numParameters = 0;

        /* calculating the number of parameters in the neural net */
        for (int i = 0; i < hiddenLayerWidths.size(); i++)
            /* number of parameters in each layer : weights + biases */
            numParameters += hiddenLayerWidths.get(i) * (i == 0 ?
                    /* number of weights in first layer */
                    initialInputLength:
                    /* number of weights in layer i */
                    hiddenLayerWidths.get(i - 1))
                    /* adding biases */
                    + hiddenLayerWidths.get(i);

        /* number of parameters in final layer : weights + biases */
        numParameters += (numClasses * hiddenLayerWidths.get(hiddenLayerWidths.size() - 1)) + numClasses;
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
            System.arraycopy(initialParameters, used, layerBiases, 0, width);
            used += width;
            /* Adding layer to list of layers */
           this.layers.add(new SingleLayer(width, layerWeights, layerBiases, activationFunctions.get(i)));
        }
        /* Initializing final layer */
        int lastLayerWidth = hiddenLayerWidths.get(hiddenLayerWidths.size() - 1);
        double[][] layerWeights = new double[numClasses][lastLayerWidth];
        double[] layerBiases = new double[numClasses];
        /*  Initializing weights for final layer */
        for (int j = 0; j < this.numClasses; j++) { /* initializing weights for neuron j in final layer*/
            for (int k = 0; k < lastLayerWidth; k++) {
                layerWeights[j][k] = initialParameters[used + (j * lastLayerWidth) + k];
            }
        }
        used += numClasses * hiddenLayerWidths.get(hiddenLayerWidths.size() - 1);
        /* Initializing biases for final layer */
        System.arraycopy(initialParameters, used, layerBiases, 0, numClasses);
        /* Adding final layer to list of layers */
        this.layers.add(new SingleLayer(numClasses, layerWeights, layerBiases,
                activationFunctions.get(activationFunctions.size() - 1)));
    }

    /**
     * forwardPass - performs a forward pass through the neural network
     *
     * @param inputs : inputs to the neural network
     * @return : outputs of the neural network
     */
    public double[] forwardPass(double[] inputs) {
        /* Array to store the activations we'll need later for gradient descent */
        /* deep copy of inputs to first set of activations */
        this.activations[0] = Arrays.copyOf(inputs, inputs.length); /* first activations are inputs */
        int counter = 1;
        for (SingleLayer layer : this.layers) {
            inputs = layer.forwardPass(inputs);
            this.activations[counter++] = inputs;
        }
        return inputs;
    }

    /**
     * crossEntropyLoss - computes the cross entropy loss of the neural network
     *
     * @param predictions : outputs of teh final output layers before SoftMax
     * @return : cross entropy loss of the neural network
     */
    public double crossEntropyLoss(double[] predictions, double[] labels) {
        /* we apply softmax to the predictions before calculating loss */
        double[] softMaxPredictions = Activation.softmax(predictions);
        double loss = 0;
        for (int i = 0; i < predictions.length; i++) loss += -Math.log(softMaxPredictions[i]) * labels[i];
        return loss;
    }

    /**
     * train - performs gradient descent on the neural network
     *
     * @param inputs : inputs to the neural network
     * @return : cross entropy loss of the neural network over training
     */

    public double[] train(double[][] inputs, double[][]correctOutputs, double learningRate){
        /* ensure inputs all have same length */
        for (double[] input : inputs) assert input.length == inputs[0].length : "All inputs must have same length";
        /* ensure correctOutputs all have same length */
        for (double[] correctOutput : correctOutputs)
            assert correctOutput.length == correctOutputs[0].length : "All correctOutputs must have same length";

        /* array to keep track of loss over training */
        double[] loss = new double[inputs.length];
        /* first we loop through all the inputs and get what the model predicts */
        for (int i = 0; i < inputs.length; i++) {
            double[] predictions = this.forwardPass(inputs[i]);
            /* now we get predictions : softmax is done within the loss function */
            loss[i] = this.crossEntropyLoss(predictions, correctOutputs[i]);
            /* now we compute the gradient */
            double[][][] gradient = this.computeGradient(correctOutputs[i], predictions);
            /* now we update the parameters */
            this.updateParameters(gradient, learningRate);
            }
        return loss;
        }

    /**
     * updateParameters - updates the parameters of the neural network
     * @param gradient : gradient of the neural network
     *                 gradient[i][j][k] = partial derivative of loss w.r.t. weight connecting neuron k in layer i-1
     *                 to neuron j in layer i.
     *                 gradient[i][j][weights[i][j].length] = partial derivative of loss w.r.t. bias of neuron j in
     *                 layer i
     * @param learningRate : learning rate for gradient descent
     */
    public void updateParameters(double[][][] gradient, double learningRate) {
        for (int i = 0; i < this.layers.size(); i++) {
            SingleLayer layer = this.layers.get(i);
            for (int j = 0; j < layer.numNeurons; j++) {
                for (int k = 0; k < layer.weights[j].length; k++) {
                    layer.weights[j][k] -= learningRate * gradient[i][j][k];
                }
                layer.biases[j] -= learningRate * gradient[i][j][layer.weights[j].length];
            }
        }
    }
    /* temp function to get derivative of ReLu - will make this more modular later */
    public double reluDerivative(double x){
        if (x > 0) return 1;
        else return 0;
    }

    /** computeGradient - computes the gradient of the neural network
     *
     * @param labels : labels for the inputs
     * @return : gradient of the neural network
     */
    public double[][][] computeGradient(double[] labels, double[] predictions) {
        double[][][] partials = new double[this.depth][][];
        double[][] preActivationPartials = new double[this.depth][];

        /* ----------------------- calculating gradient for last layer -----------------------------------------------*/
        int lastLayerIndex = this.layers.size() - 1;
        int lastLayerSize = this.layers.get(this.layers.size() - 1).numNeurons;
        int secondLastLayerSize = this.layers.get(lastLayerIndex - 1).numNeurons;

        /* Initialize the array for the last layer in partials */
        partials[lastLayerIndex] = new double[lastLayerSize][secondLastLayerSize + 1]; // +1 for biases
        preActivationPartials[lastLayerIndex] = new double[lastLayerSize];

        for (int i = 0; i < lastLayerSize; i++) {
            double ithSemiPartial = predictions[i] - labels[i];
            preActivationPartials[lastLayerIndex][i] = ithSemiPartial;
            for (int j = 0; j < secondLastLayerSize; j++) {
                partials[lastLayerIndex][i][j] = ithSemiPartial  *
                            this.activations[this.activations.length - 3][j]; /*-2 to get to layer and -1 to account
                             for fact that activations start for input layer*/
            }
            /* Gradient for bias */
            partials[lastLayerIndex][i][secondLastLayerSize] = ithSemiPartial;
        }


        /* ----------------------- remaining layers ------------------------------------------------------------------*/
        for (int i = this.layers.size() - 2; i >= 0; i--) {
            int layerSize = this.layers.get(i).numNeurons;
            int nextLayerSize = this.layers.get(i + 1).numNeurons;
            int previousLayerSize = i == 0? this.activations[0].length : this.layers.get(i - 1).numNeurons;

            /* Initialize the array for the ith layer in partials */
            partials[i] = new double[layerSize][previousLayerSize + 1]; // +1 for biases
            preActivationPartials[i] = new double[layerSize];

            for (int j = 0; j < layerSize; j++) {
                double ithSemiPartial = 0;
                for (int k = 0; k < nextLayerSize; k++) {
                    ithSemiPartial += preActivationPartials[i + 1][k] * this.layers.get(i + 1).weights[k][j];
                }
                preActivationPartials[i][j] = ithSemiPartial;
                ithSemiPartial *= reluDerivative(this.activations[i + 1][j]); /* +1 to account for fact that activations
                start from input later*/
                for (int k = 0; k < previousLayerSize; k++) {
                     partials[i][j][k] = ithSemiPartial * this.activations[i][k];
                }
                /* Gradient for bias */
                partials[i][j][previousLayerSize] = ithSemiPartial;
            }
        }
        return partials;
    }




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
     * Private inner class the represents each neuron layer in the neural network by its layer and index in that layer.
     */
    public static class Neuron {
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

        /**
         * equals - checks if two neurons are equal. Neurons are equal if they have the same layer and depth.
         * @param other : other neuron
         */
        public boolean equals(Neuron other) {
            return this.layer == other.layer && this.depth == other.depth;
        }
    }

    /**
     * Returns the neurons in the neural network
     *
     * @return : the neurons in the neural network
     */
    public Neuron[] getNeurons() {
        /* calculating number of neurons */
        int numNeurons = 0;
        for (int i = 0; i < this.depth; i++) {
            numNeurons += this.getLayerWidth(i);
        }
        /* putting all neurons into array */
        int used = 0;
        Neuron[] neurons = new Neuron[numNeurons];
        for (int i = 0; i < this.depth; i++) {
            for (int j = 0; j < this.getLayerWidth(i); j++) {
                neurons[used + j] = new Neuron(i, j, this);
            }
            used += this.getLayerWidth(i);
        }
        return neurons;
    }
}

