package reu.el.phcorrespondence;

/**
 * Various activation functions for a neural network
 * Author : Reuel Williams
 * */
public class Activation {
    /* ReLu activation function */
    public static ActivationFunction ReLu = input -> {
        if (input < 0) return 0;
        return input;
    };

    public static ActivationFunction softMax = input -> {
        double[] output = new double[input.length];
        double sum = 0;
        /* exponentiating and summing */
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i]);
            sum += output[i];
        }
        /* normalizing */
        for(int i = 0; i< output.length; i++)
            output[i] /= sum;
        return output;
    };


    /* Sigmoid activation function */
    public static double sigmoid(double input) {
        return 1 / (1 + Math.exp(-input));
    }

    /* tanh activation function */
    public static double tanh(double input) {
        return Math.tanh(input);
    }

    /* softmax activation function */
    public static double[] softmax(double[] input) {
        double[] output = new double[input.length];
        double sum = 0;
        /* exponentiating and summing */
        for (int i = 0; i < input.length; i++) {
            output[i] = Math.exp(input[i]);
            sum += output[i];
        }
        /* normalizing */
        for(int i = 0; i< output.length; i++)
            output[i] /= sum;
        return output;
    }
}
