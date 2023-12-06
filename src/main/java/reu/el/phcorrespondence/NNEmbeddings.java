//package reu.el.phcorrespondence;
//
//import edu.stanford.math.plex4.metric.interfaces.AbstractSearchableMetricSpace;
//import gnu.trove.TIntHashSet;
//
//import java.util.ArrayList;
//import java.util.List;
//
///**
// * Class designed to form filtered abstract simplicial complexes from the neurons of a neural network:
// *
// * Firstly we represent a neural network as a weighted, directed acyclic graph, Γ = (V(Γ), E, W) where V(Γ) is the set of
// * neurons in the neural network, E(Γ) is the set of edges in the neural network, and W is the weight function on the edges.
// * The simplicial complex, R_β, where, β ∈ R_{≥ 0} is defined such that for k nodes n_0, ..., n_k in the neural net,
// * σ = [n_0, ..., n_k] is in R_alpha if and only if for all i, j, d(n_i, n_j) ≤ β, where d is the semi metric on the
// * neural network detailed below:
// *
// * Let P_{n_,n_j} be the set of all paths from n_i to n_j in the neural network where a path is a finite sequence of
// * edges such that the end of one edge is the beginning of the next. Now we defined the weight w_p of a path p as follows:
// *
// * w_p = ∑_{e \in p} |w_e|,
// *
// * where w_e is the weight of edge e. Lastly let f : R_{≥ 0} → R_{≥ 0} be a monotonically
// * decreasing function such that :
// * f(0) = ∞ and f(∞) = 0.
// *
// * Then we define the distance between two nodes n_i and n_j as follows:
// *
// * ∑_{p \in P_{n_i,n_j}} f(w_p)
// *
// */
//public class NNEmbeddings {
//
//    public static void main(String[] args) {
//        /* creating neural network with weights initialized randomly to all have absolute value less than 1 */
//        ParameterSampler randomSampler = new PointSampler.randomSampler(1.0).getSampler();
//
//        /* Activation functions */
//        List<ActivationFunction> activationFunctions = new ArrayList<ActivationFunction>();
//        activationFunctions.add(Activation.ReLu);
//        activationFunctions.add(Activation.ReLu);
//        activationFunctions.add(Activation.ReLu);
//
//        /* Layer sizes */
//        List<Integer> layerSizes = new ArrayList<Integer>();
//        int initialInputLength = 784;
//        int numClasses = 10;
//
//        SimpleConnectedNeuralNet neuralNet = new SimpleConnectedNeuralNet(randomSampler, activationFunctions, layerSizes, initialInputLength, numClasses);
//
//        List<SimpleConnectedNeuralNet.Neuron> neurons = neuralNet.getNeurons();
//
//        // Convert your data and metric into a format JavaPlex understands
//        AbstractSearchableMetricSpace<SimpleConnectedNeuralNet.Neuron> semiMetricSpace = new AbstractSearchableMetricSpace<SimpleConnectedNeuralNet.Neuron>() {
//
//            /**
//             * Returns the distance between two points in the metric space using the semi metric as described on the
//             * top of this file.
//             * @param n1 : First neuron
//             * @param n2: Second neuron
//             * @return
//             */
//            @Override
//            public double distance(SimpleConnectedNeuralNet.Neuron n1, SimpleConnectedNeuralNet.Neuron n2) {
//                /* distance only depends on the position of the neurons in the NN since it is dense */
//
//                /* neurons in the same layer are not connected*/
//                if (n1.layer == n2.layer) return Double.POSITIVE_INFINITY;
//
//                else if (n1.layer > n2.layer) {
//                    /* swap n1 and n2 */
//                    SimpleConnectedNeuralNet.Neuron temp = n1;
//                    n1 = n2;
//                    n2 = temp;
//                }
//                /* we calculate w_p for each path, sum, then take multiplicative inverse */
//                int numPaths = 1;
//
//                for (int i = n1.layer; i < n2.layer; i++) {
//                    numPaths *= n1.net.getLayerWidth(i);
//                }
////                for(int i = 0; i < numPaths; i++){
////                    /* calculate w_p for each path */
////                }
//            return 0;
//            }
//
//            @Override
//            public double distance(int i, int i1) {
//                return 0;
//            }
//
//            @Override
//            public int size() {
//                return 0;
//            }
//
//            @Override
//            public SimpleConnectedNeuralNet.Neuron getPoint(int i) {
//                return null;
//            }
//
//            @Override
//            public SimpleConnectedNeuralNet.Neuron[] getPoints() {
//                return new SimpleConnectedNeuralNet.Neuron[0];
//            }
//
//            @Override
//            public int getNearestPointIndex(SimpleConnectedNeuralNet.Neuron neuron) {
//                return 0;
//            }
//
//            @Override
//            public TIntHashSet getOpenNeighborhood(SimpleConnectedNeuralNet.Neuron neuron, double v) {
//                return null;
//            }
//
//            @Override
//            public TIntHashSet getClosedNeighborhood(SimpleConnectedNeuralNet.Neuron neuron, double v) {
//                return null;
//            }
//
//            @Override
//            public TIntHashSet getKNearestNeighbors(SimpleConnectedNeuralNet.Neuron neuron, int i) {
//                return null;
//            }
//
//
//        };
//    }
//}
