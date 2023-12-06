package reu.el.phcorrespondence;

import edu.stanford.math.plex4.homology.barcodes.BarcodeCollection;

import java.io.IOException;

public class run {

    /**
     * Calculates the average distance between pairs of embeddings with given labels.
     *
     * @param MNISTEmbeddings An Embeddings object containing the embeddings and labels.
     * @param first The label of the first type of embedding.
     * @param second The label of the second type of embedding.
     * @return The average distance between the pairs, or -1 if no such pairs exist.
     */
    private double averageDistance(Embeddings MNISTEmbeddings, int first, int second) {
        int firstToCheck;
        int secondToCheck;
        boolean noPairsFoundYet = false;
        double averageDistance = 0;
        double[][] embeddings = MNISTEmbeddings.getEmbeddings();
        int[] labels = MNISTEmbeddings.getLabels();
        int check = first == second? 1 : 2;

        for (int i = 0; i < check; i++) {

            /* we first check the average distance of pairs of the form (first, second) where second has index
            * higher that first -- then we check pairs of the form (second, first) where first has index higher than
            * second.
            *
            * The above is only done if first != second */
            firstToCheck = i == 0 ? first : second;
            secondToCheck = i == 0 ? second : first;

            double distance = 0;
            int count = 0;
            for (int j = 0; j < embeddings.length; j++) {
                if (labels[j] == firstToCheck) {
                    for (int k = j + 1; k < embeddings.length; k++) {
                        if (labels[k] == secondToCheck) {
                            distance += MNISTEmbeddings.distance(j, k);
                            count++;
                        }
                    }
                }
            }
            /* return -1 if no pairs of embeddings with labels (first, second) were found */
            if (count == 0) {
                if (!noPairsFoundYet)  {
                    noPairsFoundYet = true;
                    continue;
                }
                return -1;
            }
            averageDistance += distance / count;

        }
        return averageDistance / check;
    }

    public static void main(String[] args) throws IOException {
        /* pull photos and get embeddings */
        String pathToPhotos = "src/main/resources/MNIST-TEST/0-5-small";
        Embeddings MNISTEmbeddings = new Embeddings(pathToPhotos);
        int dimensionReduction = 0;

        /* collapsing coordinates to see if this makes things cleaner */
        MNISTEmbeddings.collapseRandomCoordinates(dimensionReduction);

        /* calculate average distance between pairs of embeddings with labels 0 and 5 */
        int first = 2;
        int second = 3;

        run r = new run();

        double averageDistance = r.averageDistance(MNISTEmbeddings, first, second);
        System.out.println("Average distance between pairs of embeddings with labels " + first + " and "
                + second + ": " + averageDistance);

        /* calculate barcodes */
        double maxFiltrationValue = 2000;
        int maxDimension = 3;
        int torsion = 2;

        BarcodeCollection<Double> MNISTBarcode = MNISTEmbeddings.computeVRPersistentHomology(maxFiltrationValue,
                maxDimension, torsion);
        System.out.println(MNISTBarcode);
    }
}
