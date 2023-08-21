import org.junit.jupiter.api.Test;
import reu.el.phcorrespondence.embeddings;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;


public class embeddingsTest {
    private String pathToPhotos; // Path to parent directory of images.

    public embeddingsTest(String s) {
        this.pathToPhotos = s;
    }

    /**
     * Creates an Image object given an array of grey scale values.
     *
     * @return image : Image object corresponding to grey scale valyes.
     * @throws IOException
     * @ param greyScaleValues : Array of grey scale values.
     */
    private static BufferedImage createImage(double[] greyScaleValues) {
        int width = (int) Math.sqrt(greyScaleValues.length);
        int height = (int) Math.sqrt(greyScaleValues.length);
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        int index = 0;
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                int greyScaleValue = (int) greyScaleValues[index];
                int rgb = (greyScaleValue << 16) | (greyScaleValue << 8) | greyScaleValue;
                image.setRGB(i, j, rgb);
                index++;
            }
        }
        return image;
    }

    /**
     * Gets an array of images give the path to the parent directory of the images.
     *
     * @param pathToPhotos : Path to parent directory of images.
     *                     Images must be in a subdirectory of the parent directory.
     * @return images : Array of images.
     */
    private static BufferedImage[] getImages(String pathToPhotos) throws IOException {
        File[] photos = new File(pathToPhotos).listFiles();
        BufferedImage[] images = new BufferedImage[photos.length];
        for (int i = 0; i < photos.length; i++) {
            images[i] = ImageIO.read(photos[i]);
        }
        return images;
    }

    /**
     * Tests the embeddings class.
     * @return : True if test passes, false otherwise.
     */
    public boolean test() throws IOException {
         embeddings testEmbeddings = new embeddings(this.pathToPhotos);
        double[][] testEmbeddingsArray = testEmbeddings.getEmbeddings();
        BufferedImage[] testImages = new BufferedImage[testEmbeddingsArray.length];
        for (int i = 0; i < testEmbeddingsArray.length; i++) {
            /* for each embedding reconstruct photo */
            testImages[i] = createImage(testEmbeddingsArray[i]);
        }
        /* test passes if image is reconstructed properly */
        return getImages("src/test/resources/testPhotos").equals(testImages);
    }
}



