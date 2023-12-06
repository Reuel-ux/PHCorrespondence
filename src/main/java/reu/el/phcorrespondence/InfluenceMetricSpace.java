package reu.el.phcorrespondence;

import edu.stanford.math.plex4.metric.impl.ObjectSearchableFiniteMetricSpace;

import java.util.ArrayList;
import java.util.List;


/** DAG
 * This class implements a metric space on the neurons of a neural network with the distance functions r : Γ × Γ → R≥0
 * defined on the vertex set {n_0 , ..., n_K} = V(Γ) of a neural network Γ as follows:
 *
 *                  r(_i,n_j) = 0 iff i = j and r(n_i, n_j) = ∞ iff there is no path from n_i → n_j in Γ.
 *
 */
public class InfluenceMetricSpace extends ObjectSearchableFiniteMetricSpace<SimpleConnectedNeuralNet.Neuron> {
    private MetricFunction f; /* the monotonically decreasing function on which teh semi metric depends */

    /**
     * returns n paths from n_i to n_j in the neural network where a path is a finite sequence of edges.
     * Each path is represented as a list of Integers where each integer is the index of a neuron in the neural network.
     * so that (path(i - 1), path(i)) is an edge in the neural network.
     *
     * @param n : number of paths to return
     * @return
     */
    private ArrayList<ArrayList<Integer>> randomPaths(int n, SimpleConnectedNeuralNet.Neuron precedingNeuron, int pathLength
    ) {

        /* instance of simply connected neuron net */
        SimpleConnectedNeuralNet neuralNet = precedingNeuron.net;
        int pathsCreated = 0;
        ArrayList<ArrayList<Integer>> paths = new ArrayList<>();
        while (pathsCreated < n) {
            ArrayList<Integer> path = new ArrayList<Integer>();
            for (int i = 0; i < pathLength; i++) {
                int randomNeuron = (int) (Math.random() * neuralNet.getLayerWidth(precedingNeuron.layer + i + 1));
                path.add(randomNeuron);
            }
            paths.add(path);
        }
        return paths;
    }
    /** performs matrix multiplication on two matrices
     *
     * @param first
     * @param second
     * @return : the product of the two matrices
     */
    private double[][] AbsoluteValueMatrixMultiplication(double[][] first, double[][] second) {
        int firstRows = first.length;
        int firstColumns = first[0].length;
        int secondRows = second.length;
        int secondColumns = second[0].length;
        if (firstColumns != secondRows) {
            throw new IllegalArgumentException("Matrices don't match: " + firstColumns + " != " + secondRows);
        }
        double[][] result = new double[firstRows][secondColumns];
        for (int i = 0; i < firstRows; i++) {
            for (int j = 0; j < secondColumns; j++) {
                for (int k = 0; k < firstColumns; k++) {
                    result[i][j] += Math.abs(first[i][k] * second[k][j]);
                }
            }
        }
        return result;
    }

    /** makes all the entries in a matrix positive
     *
     * @param matrix
     * @return : the matrix with all positive entries
     */

    private double[][] matrixAbsoluteValue(double[][] matrix) {
        for (int i= 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[0].length; j++)
                matrix[i][j] = Math.abs(matrix[i][j]);
        }
        return matrix;
    }

    public InfluenceMetricSpace(SimpleConnectedNeuralNet.Neuron[] var) {
        super(var);
    }

    @Override
    public double distance(SimpleConnectedNeuralNet.Neuron neuron, SimpleConnectedNeuralNet.Neuron t1) {
        /* if they are the same distance is zero */
        if (neuron.equals(t1)) return 0;

        /* if they are in the same layer then they are infinitely far away */
        if (neuron.layer == t1.layer) return Double.POSITIVE_INFINITY;


        /* ordering the neuron */
        SimpleConnectedNeuralNet.Neuron precedingNeuron = neuron.layer > t1.layer ? neuron : t1;
        SimpleConnectedNeuralNet.Neuron followingNeuron = neuron.layer > t1.layer ? t1 : neuron;

        /* instance of simply connected neuron net */
        SimpleConnectedNeuralNet neuralNet = precedingNeuron.net;

        /* we do matrix multiplication of the weight matrices between the layers */
        int firstMatrixRows = precedingNeuron.layer;
        double[][] resultingSums = matrixAbsoluteValue(neuralNet.getLayerWeights(firstMatrixRows));
        for (int i = precedingNeuron.layer; i < followingNeuron.layer; i++) {
            double[][] nextLayerWeights = neuralNet.getLayerWeights(firstMatrixRows + i );
            resultingSums = AbsoluteValueMatrixMultiplication(nextLayerWeights, resultingSums);

        }

        /* now that we have the sums we can take apply the inverting metric function */
        double sum = resultingSums[precedingNeuron.depth][followingNeuron.depth];
        if (sum == 0) return Double.POSITIVE_INFINITY;
        return 1/sum;

    }

}


