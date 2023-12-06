import edu.stanford.math.plex4.homology.barcodes.BarcodeCollection;
import reu.el.phcorrespondence.Embeddings;

import java.io.IOException;

public class testRun {
    public static void main(String[] args) throws IOException {
        Embeddings testEmbeddings = new Embeddings("/Users/reuelwilliams/IdeaProjects/PHCorrespondence/src/main/resources/MNIST-0");
        double maxFiltrationValue = 1000.0;
        BarcodeCollection barcodeCollection = testEmbeddings.computeVRPersistentHomology(maxFiltrationValue,3,2);
        System.out.println(barcodeCollection);
    }
}
