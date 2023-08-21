package reu.el.phcorrespondence;
import java.util.Random;

/**
 *  PointSampler - Generate points on various topological spaces with different types of probability distributions.
 * Author : Reuel Williams
 */
public class PointSampler {

    /**
     * Generates a point sampled randomly with a uniform distribution but with absolute value
     * lower than some threshold
     */
    public static class randomSampler {
        private double threshold; /* Max absolute value of coordinate */

        public randomSampler(double threshold) {
            this.threshold = threshold;
        }

        /**
         * Generates a point sampled randomly with a uniform distribution but with absolute value
         * @return : A ParameterSampler object that can be used to generate points
         */
        public ParameterSampler getSampler() {
            return new ParameterSampler() {
                @Override
                public double[] sampleParameters(int dimension) {
                    double[] parameters = new double[dimension];
                    Random random = new Random();
                    for (int i = 0; i < dimension; i++) {
                        parameters[i] = random.nextDouble() * threshold;
                        if (random.nextBoolean()) parameters[i] *= -1;
                    }
                    return parameters;
                }
            };
        }
    }

}

