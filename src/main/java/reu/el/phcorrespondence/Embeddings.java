package reu.el.phcorrespondence;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;

import edu.stanford.math.plex4.homology.interfaces.AbstractPersistenceAlgorithm;
import edu.stanford.math.plex4.metric.impl.EuclideanMetricSpace;
import edu.stanford.math.plex4.api.*;
import edu.stanford.math.plex4.streams.impl.VietorisRipsStream;
import lombok.Data;
import edu.stanford.math.plex4.homology.barcodes.BarcodeCollection;




/**
 *  Class then generates embeddings into a vector space given some data set. Possible types of data include images, text,
 *  and audio.
 *  Also allows for the computation of the distance between two embeddings, and calculations related to the Persistent
 *  Homology of the embeddings.
 */
@Data
public class Embeddings {
    private double[][] Embeddings; // Embeddings of the data set.
    private int[] labels; // Labels of the data set.
    int IMAGE_TYPE_OFFSET = 5; // Offset to get the first number in the file name.

    /**
     * Private helper function to a colored pixel to grey scale.
     *
     * @param pixel : Pixel to be converted to grey scale.
     * @return : Grey scale value of the pixel.
     */
    private static int pixelToGreyScale(int pixel) {
        // extracting rgb values
        int red = (int) ((pixel >> 16) & 0xff);
        int green = (int) ((pixel >> 8) & 0xff);
        int blue = (int) (pixel & 0xff);
        // converting to grey scale
        return (int) ((0.21 * red) + (0.72 * green) + (0.07 * blue));
    }

    /**
     * Private helper function to convert an image to a vector where each coordinate is the rbg value of the image.
     * Image is converted to grey scale before being converted to a vector.
     * Image must be a square if not it is cropped to be as big as it can be while still being a square from the bottom
     * right.
     * @param photo : Image to be converted to a vector.
     * @return : Vector representation of the image.
     */
    private static double[] imageToVector(File photo) throws IOException {
        BufferedImage image = ImageIO.read(photo);
        int width = image.getWidth();
        int height = image.getHeight();

        if (width != height) {
            int maxDim = Math.min(width, height);
            width = maxDim;
            height = maxDim;
        }
        double[] vector = new double[width * height];
        int index = 0;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                vector[index++] = pixelToGreyScale(image.getRGB(i, j));
            }
        }
        return vector;
    }

    /** private helper function to check for DS_Store file
     *
     * @param files : array of files to check
     * @return : true if there is a DS_Store file, false otherwise
     */
    private static boolean isDSStore(File[] files) {
        for (File file : files) {
            if (file.getName().equals(".DS_Store")) return true;
        }
        return false;
    }

    /**
     * Constructor for embeddings class given photos. Photos are embedding with labels.
     *
     * @param pathToPhotos : File containing photos to be embedded.
     */

    public Embeddings(String pathToPhotos) {
        File dir = new File(pathToPhotos);
        File[] photos;
        if (!dir.isDirectory()) throw new AssertionError("Path to photos is not a directory");

        /* make sure photos exists and check for DS_Store file */
        photos = dir.listFiles();
        if (photos == null) throw new AssertionError("No photos found in directory");
        boolean isDSStore = isDSStore(photos);
        /* -1 because of .DS_Store file */
        this.Embeddings = isDSStore? new double[photos.length - 1][] : new double[photos.length][];
        this.labels = isDSStore? new int[photos.length - 1] : new int[photos.length];

        /* embed photos */
        int index = 0;
        for (File photo : photos) {
            if (photo.getName().equals(".DS_Store")) continue;
            try {
                this.Embeddings[index] = imageToVector(photo);
                char label = photo.getName().charAt(photo.getName().length() - IMAGE_TYPE_OFFSET); /* last character of file name is label */
                this.labels[index++] = Character.getNumericValue(label);
            } catch (IOException e) {
                // Log the error
                System.out.println("Couldn't embed photo: " + e);
            }
        }
    }

    /**
     * Constructor given a point cloud.
     */
    public Embeddings(double[][] pointCloud) {
        this.Embeddings = pointCloud;
    }

    /**
     * Returns string representation of embeddings give its index.
     * @param i : index of embedding to be printed.
     * @return : String representation of the embedding.
     */
    public String embeddingToString(int i) {
        StringBuilder sb = new StringBuilder();
        sb.append("(");
        for (int j = 0; j < this.Embeddings[i].length; j++) {
            sb.append(this.Embeddings[i][j]);
            if (j != this.Embeddings[i].length - 1) sb.append(", ");
        }
        sb.append(")");
        return sb.toString();
    }

    /**
     * Returns string representation of labels
     * @return : String representation of labels.
     */
    public String labelsToString() {
        StringBuilder sb = new StringBuilder();
        sb.append("(");
        for (int i = 0; i < this.labels.length; i++) {
            sb.append(this.labels[i]);
            if (i != this.labels.length - 1) sb.append(", ");
        }
        sb.append(")");
        return sb.toString();
    }
    /**
     * Computes the euclidean distance between two embeddings.
     * @param i : index of first emeddding.
     * @param j : index of second embedding.
     */
    public double distance(int i, int j) {
        double distance = 0;
        for (int k = 0; k < this.Embeddings[i].length; k++) {
            distance += Math.pow(this.Embeddings[i][k] - this.Embeddings[j][k], 2);
        }
        return Math.sqrt(distance);
    }

    /** Collapses <n> random coordinates reducing the dimension of the emdeeings from m to m - n
     * @param n : number of coordinates to collapse
     */
    public void collapseRandomCoordinates(int n) {
        if (n == 0) return;
        int m = this.Embeddings[0].length;

        /* number of coordinates to remove must be less that dimension of ambient space */
        if (n > m) throw new AssertionError("Number of coordinates to remove must be less than dimension " +
                "of ambient space");

        /* generating coordinates to remove*/
        HashSet<Integer> randomCoordinates = new HashSet<>();
        while (randomCoordinates.size() < n) {
            int randomCoordinate = (int) (Math.random() * m);
            randomCoordinates.add(randomCoordinate);
        }

        for (int i = 0; i < this.Embeddings.length; i++) {
            double[] newEmbedding = new double[m - n];
            int index = 0;
            for (int j = 0; j < m; j++) {
                if (!randomCoordinates.contains(j)) {
                    newEmbedding[index++] = this.Embeddings[i][j];
                }
            }
            this.Embeddings[i] = newEmbedding;
        }
    }

    /** Creates a Filtered Vietoris Rips Complex with the embeddings and computes its PH barcode
     * @param maxFiltrationValue : Maximum filtration value for a simplex to be created.
     * @param maxDimension : Maximum dimension of a simplex to be created.
     * @param torsion : Order n of the finite field Z_n over which the persistence homology is calculated.
     * @return : PH barcode of the filtered Vietoris Rips Complex.
     */
    public BarcodeCollection computeVRPersistentHomology(double maxFiltrationValue, int maxDimension, int torsion){
        /* R^n where n is the number of embeddings */
        EuclideanMetricSpace dataEmbeddings = new EuclideanMetricSpace(this.Embeddings);
        /* creates filtered VR complex that can detect cycles up to dimension <maxDimension> */
        VietorisRipsStream stream = new VietorisRipsStream(dataEmbeddings, maxFiltrationValue, maxDimension + 1);
        stream.finalizeStream();

        AbstractPersistenceAlgorithm persistence = Plex4.getModularSimplicialAlgorithm(maxDimension + 1, torsion);
        return persistence.computeIntervals(stream);
    }

}