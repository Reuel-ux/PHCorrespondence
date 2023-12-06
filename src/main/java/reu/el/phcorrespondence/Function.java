package reu.el.phcorrespondence;

public class Function {

    public static MetricFunction inverse = input -> {
        if (input == 0) return Double.POSITIVE_INFINITY;
        if (input == Double.POSITIVE_INFINITY) return 0;
        return 1 / input;
    };
}
