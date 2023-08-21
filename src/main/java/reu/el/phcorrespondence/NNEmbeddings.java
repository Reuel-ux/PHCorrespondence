package reu.el.phcorrespondence;

import edu.stanford.math.plex4.metric.interfaces.AbstractSearchableMetricSpace;
import gnu.trove.TIntHashSet;

import java.util.ArrayList;
import java.util.List;

/**
 * Class designed to form filtered abstract simplicial complexes from the neurons of a neural network. The simplicial complex,
 * R_alpha, where, alpha is in R is defined such that for k nodes n_0, ..., n_k in the neural net, sigma = [n_0, ..., n_k] is in R_alpha if and only if
 * for all i, j, d(n_i, n_j) <= alpha, where d is the distance function on the neural network. Defined to be
 */
public class NNEmbeddings {

    public static void main(String[] args) {
        /* creating neural network with weights initialized randomly to all have absolute value less than 1 */
        ParameterSampler randomSampler = new PointSampler.randomSampler(1.0).getSampler();

        /* Activation functions */
        List<ActivationFunction> activationFunctions = new ArrayList<ActivationFunction>();
        activationFunctions.add(Activation.ReLu);
        activationFunctions.add(Activation.ReLu);

        /* Layer sizes */
        List<Integer> layerSizes = new ArrayList<Integer>();
        int initialInputLength = 784;
        int numClasses = 10;

        SimpleConnectedNeuralNet neuralNet = new SimpleConnectedNeuralNet(randomSampler, activationFunctions, layerSizes, initialInputLength, numClasses);

        List<SimpleConnectedNeuralNet.Neuron> neurons = neuralNet.getNeurons();

        // Convert your data and metric into a format JavaPlex understands
        AbstractSearchableMetricSpace<SimpleConnectedNeuralNet.Neuron> semiMetricSpace = new AbstractSearchableMetricSpace<SimpleConnectedNeuralNet.Neuron>(neurons) {

            /**
             * Returns the distance between two points in the metric space using the semi metric as described on the
             * top of this file.
             * @param n1 : First neuron
             * @param n2: Second neuron
             * @return
             */
            @Override
            public double distance(SimpleConnectedNeuralNet.Neuron n1, SimpleConnectedNeuralNet.Neuron n2) {
                /* distance only depends on the position of the neurons in the NN since it is dense */

                /* neurons in the same layer are not connected*/
                if (n1.layer == n2.layer) return Double.POSITIVE_INFINITY;

                else if (n1.layer > n2.layer) {
                    /* swap n1 and n2 */
                    SimpleConnectedNeuralNet.Neuron temp = n1;
                    n1 = n2;
                    n2 = temp;
                }
                /* we calculate w_p for each path, sum, then take multiplicative inverse */
                int numPaths = 1;

                for (int i = n1.layer; i < n2.layer; i++) {
                    numPaths *= n1.net.getLayerWidth(i);
                }
                for(int i = 0; i < numPaths; i++){
                    /* calculate w_p for each path */
                    double
                }

            }

            @Override
            public double distance(int i, int i1) {
                return 0;
            }

            @Override
            public int size() {
                return 0;
            }

            @Override
            public SimpleConnectedNeuralNet.Neuron getPoint(int i) {
                return null;
            }

            @Override
            public SimpleConnectedNeuralNet.Neuron[] getPoints() {
                return new SimpleConnectedNeuralNet.Neuron[0];
            }

            @Override
            public int getNearestPointIndex(SimpleConnectedNeuralNet.Neuron neuron) {
                return 0;
            }

            @Override
            public TIntHashSet getOpenNeighborhood(SimpleConnectedNeuralNet.Neuron neuron, double v) {
                return null;
            }

            @Override
            public TIntHashSet getClosedNeighborhood(SimpleConnectedNeuralNet.Neuron neuron, double v) {
                return null;
            }

            @Override
            public TIntHashSet getKNearestNeighbors(SimpleConnectedNeuralNet.Neuron neuron, int i) {
                return null;
            }

            @Override
            public double distance(MyDataObject a, MyDataObject b) {
                // Your distance function here
                return ...;
            }
        };
    }
}
