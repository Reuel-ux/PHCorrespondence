package reu.el.phcorrespondence;

import edu.stanford.math.plex4.metric.interfaces.AbstractSearchableMetricSpace;
import gnu.trove.TIntHashSet;

/** DAG
 * This class implements a metric space on the neurons of a neural network with the distance functions r : Γ × Γ → R≥0
 * defined on the vertex set {n_0 , ..., n_K} = V(Γ) of a neural network Γ as follows:
 *
 *                  r(_i,n_j) = 0 iff i = j and r(n_i, n_j) = ∞ iff there is no path from n_i → n_j in Γ.
 *
 */
public class InfluenceMetricSpace implements AbstractSearchableMetricSpace {
    @Override
    public int getNearestPointIndex(Object o) {
        return 0;
    }

    @Override
    public TIntHashSet getOpenNeighborhood(Object o, double v) {
        return null;
    }

    @Override
    public TIntHashSet getClosedNeighborhood(Object o, double v) {
        return null;
    }

    @Override
    public TIntHashSet getKNearestNeighbors(Object o, int i) {
        return null;
    }

    @Override
    public Object getPoint(int i) {
        return null;
    }

    @Override
    public Object[] getPoints() {
        return new Object[0];
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
    public double distance(Object o, Object t1) {
        return 0;
    }
}
