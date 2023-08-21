package reu.el.phcorrespondence;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

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
public class embeddings {
    private double[][] Embeddings; // Embeddings of the data set.

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
                vector[index] = pixelToGreyScale(image.getRGB(i, j));
                index++;
            }
        }
        return vector;
    }

    /**
     * Constructor for embeddings class given photos.
     *
     * @param pathToPhotos : File containing photos to be embedded.
     */

    public embeddings(String pathToPhotos) throws IOException {
        int index = 0;
        File[] photos = new File(pathToPhotos).listFiles();
        this.Embeddings = new double[photos.length][];
        for (File photo : photos) {
            try { this.Embeddings[index++] = imageToVector(photo);}
             catch (IOException e) {System.out.println("Couldn't embed photo: " + e);}
        }
    }
    /**
     * Constructor given a point cloud.
     */
    public embeddings(double[][] pointCloud) {
        this.Embeddings = pointCloud;
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