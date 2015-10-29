package com.github.davityle.ml;

import com.github.davityle.ml.writtenbyprofessor.Matrix;
import com.github.davityle.ml.writtenbyprofessor.SupervisedLearner;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;

public class Perceptron extends SupervisedLearner {

    private static final double LEARNING_RATE = .1, THRESHOLD = 0;
    private double[][] weights;
    private int perceptronCount;
    private final Random random;

    public Perceptron(Random random) {
        this.random = random;
    }

    private double acc(Matrix inputs, Matrix labels) {
        try {
            return measureAccuracy(inputs, labels, null);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return 0;
    }

    private int trn(Matrix inputs, Matrix labels, int count) {
        int epochCount = 0;
        double currentAccuracy = acc(inputs, labels);
        double m = currentAccuracy;
        while (count > 0) {
            for (int i = 0; i < weights.length; i++) {
                final double finalI = i;
                weights[i] = epoch(inputs, weights[i], weights.length != 1 ? row -> (labels.get(row, 0) == finalI ? 1d : 0d) : row -> labels.get(row, 0));
            }
            double newAcc = acc(inputs, labels);
            double diff = currentAccuracy - newAcc;
            if (diff >= -.01 && diff <= 0.01) {
                count--;
            }
            inputs.shuffle(random, labels);
            epochCount++;
            currentAccuracy = newAcc;
            if (currentAccuracy > m) {
                m = currentAccuracy;
            }
        }
        return epochCount;
    }

    private double[] epoch(Matrix inputs, double[] weights, Function<Integer, Double> target) {
        for (int i = 0; i < inputs.rows(); i++) {
            weights = calculate(inputs.row(i), target.apply(i), weights);
        }
        return weights;
    }

    private double[] calculate(double[] input, double target, double[] weights) {
        double result = neuron(input, weights);
        double diff = target - result;
        if (result != diff) {
            for (int i = 0; i < input.length; i++) {
                weights[i] = weights[i] + diff * LEARNING_RATE * input[i];
            }
        }
        return weights;
    }

    private double multZip(double[] input, double[] weights) {
        double sum = 0;
        for (int i = 0; i < input.length; i++) {
            sum += (input[i] * weights[i]);
        }
        return sum;
    }

    private int neuron(double[] input, double[] weights) {
        return multZip(input, weights) >= THRESHOLD ? 1 : 0;
    }


    @Override
    public void train(Matrix features, Matrix labels) throws Exception {
        int resultsLength = labels.getResultsLength(0);
        perceptronCount = resultsLength <= 2 ? 1 : resultsLength;
        weights = new double[perceptronCount][features.cols()];
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {
        if (perceptronCount == 1) {
            labels[0] = neuron(features, weights[0]);
        } else {
            List<Integer> positives = new ArrayList<>(perceptronCount);
            for (int i = 0; i < perceptronCount; i++) {
                int result = neuron(features, weights[i]);
                if (result > 0)
                    positives.add(Integer.valueOf(i));
            }

            labels[0] = positives.stream()
                    .sorted((i1, i2) -> (int) (multZip(features, weights[i1]) - multZip(features, weights[i2])))
                    .findFirst()
                    .orElseGet(() -> 0);
        }
    }
}
