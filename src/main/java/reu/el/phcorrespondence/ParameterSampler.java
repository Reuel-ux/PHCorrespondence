package reu.el.phcorrespondence;

/**
 *  Functional Interface to be used to pass in initial parameterization sampling in SimpleConnectedNeuralNet Object
 *
 */
public interface ParameterSampler {
    double[] sampleParameters(int dimension);
}
