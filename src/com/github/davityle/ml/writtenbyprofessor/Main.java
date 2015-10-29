package com.github.davityle.ml.writtenbyprofessor;

import com.github.davityle.ml.DecisionTree;
import com.github.davityle.ml.InstanceBasedLearner;
import com.github.davityle.ml.NeuralNet;
import com.github.davityle.ml.Perceptron;

import java.util.Random;

/**
 * This class is code supplied by my professor with some small amount of cleanup
 */
public class Main {

    public SupervisedLearner getLearner(String model, Random rand) throws Exception {
        switch (model) {
            case "baseline":
                return new BaselineLearner();
            case "perceptron":
                return new Perceptron(rand);
            case "neuralnet":
                return new NeuralNet(rand);
            case "decisiontree":
                return new DecisionTree();
            case "knn":
                return new InstanceBasedLearner();
            default:
                throw new Exception("Unrecognized model: " + model);
        }
    }

    public void run(String[] args) throws Exception {

        Random rand = new Random();

        //Parse the command line arguments
        ArgParser parser = new ArgParser(args);
        String fileName = parser.getARFF(); //File specified by the user
        String learnerName = parser.getLearner(); //Learning algorithm specified by the user
        String evalMethod = parser.getEvaluation(); //Evaluation method specified by the user
        String evalParameter = parser.getEvalParameter(); //Evaluation parameters specified by the user
        boolean printConfusionMatrix = parser.getVerbose();
        boolean normalize = parser.getNormalize();

        // Load the model
        SupervisedLearner learner = getLearner(learnerName, rand);

        // Load the ARFF file
        Matrix data = new Matrix();
        data.loadArff(fileName);
        if (normalize) {
            System.out.println("Using normalized data\n");
            data.normalize();
        }

        // Print some stats
        System.out.println();
        System.out.println("Dataset name: " + fileName);
        System.out.println("Number of instances: " + data.rows());
        System.out.println("Number of attributes: " + data.cols());
        System.out.println("Learning algorithm: " + learnerName);
        System.out.println("Evaluation method: " + evalMethod);
        System.out.println();

        switch (evalMethod) {
            case "training": {
                System.out.println("Calculating accuracy on training set...");
                Matrix features = new Matrix(data, 0, 0, data.rows(), data.cols() - 1);
                Matrix labels = new Matrix(data, 0, data.cols() - 1, data.rows(), 1);
                Matrix confusion = new Matrix();
                double startTime = System.currentTimeMillis();
                learner.train(features, labels);
                double elapsedTime = System.currentTimeMillis() - startTime;
                System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
                double accuracy = learner.measureAccuracy(features, labels, confusion);
                System.out.println("Training set accuracy: " + accuracy);
                if (printConfusionMatrix) {
                    System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
                    confusion.print();
                    System.out.println("\n");
                }
                break;
            }
            case "static": {
                Matrix testData = new Matrix();
                testData.loadArff(evalParameter);
                if (normalize)
                    testData.normalize(); // BUG! This may normalize differently from the training data. It should use the same ranges for normalization!

                System.out.println("Calculating accuracy on separate test set...");
                System.out.println("Test set name: " + evalParameter);
                System.out.println("Number of test instances: " + testData.rows());
                Matrix features = new Matrix(data, 0, 0, data.rows(), data.cols() - 1);
                Matrix labels = new Matrix(data, 0, data.cols() - 1, data.rows(), 1);
                double startTime = System.currentTimeMillis();
                learner.train(features, labels);
                double elapsedTime = System.currentTimeMillis() - startTime;
                System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
                double trainAccuracy = learner.measureAccuracy(features, labels, null);
                System.out.println("Training set accuracy: " + trainAccuracy);
                Matrix testFeatures = new Matrix(testData, 0, 0, testData.rows(), testData.cols() - 1);
                Matrix testLabels = new Matrix(testData, 0, testData.cols() - 1, testData.rows(), 1);
                Matrix confusion = new Matrix();
                double testAccuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
                System.out.println("Test set accuracy: " + testAccuracy);
                if (printConfusionMatrix) {
                    System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
                    confusion.print();
                    System.out.println("\n");
                }
                break;
            }
            case "random": {
                System.out.println("Calculating accuracy on a random hold-out set...");
                double trainPercent = Double.parseDouble(evalParameter);
                if (trainPercent < 0 || trainPercent > 1)
                    throw new Exception("Percentage for random evaluation must be between 0 and 1");
                System.out.println("Percentage used for training: " + trainPercent);
                System.out.println("Percentage used for testing: " + (1 - trainPercent));
                data.shuffle(rand);
                int trainSize = (int) (trainPercent * data.rows());
                Matrix trainFeatures = new Matrix(data, 0, 0, trainSize, data.cols() - 1);
                Matrix trainLabels = new Matrix(data, 0, data.cols() - 1, trainSize, 1);
                Matrix testFeatures = new Matrix(data, trainSize, 0, data.rows() - trainSize, data.cols() - 1);
                Matrix testLabels = new Matrix(data, trainSize, data.cols() - 1, data.rows() - trainSize, 1);
                double startTime = System.currentTimeMillis();
                learner.train(trainFeatures, trainLabels);
                double elapsedTime = System.currentTimeMillis() - startTime;
                System.out.println("Time to train (in seconds): " + elapsedTime / 1000.0);
                double trainAccuracy = learner.measureAccuracy(trainFeatures, trainLabels, null);
                System.out.println("Training set accuracy: " + trainAccuracy);
                Matrix confusion = new Matrix();
                double testAccuracy = learner.measureAccuracy(testFeatures, testLabels, confusion);
                System.out.println("Test set accuracy: " + testAccuracy);
                if (printConfusionMatrix) {
                    System.out.println("\nConfusion matrix: (Row=target value, Col=predicted value)");
                    confusion.print();
                    System.out.println("\n");
                }
                break;
            }
            case "cross": {
                System.out.println("Calculating accuracy using cross-validation...");
                int folds = Integer.parseInt(evalParameter);
                if (folds <= 0)
                    throw new Exception("Number of folds must be greater than 0");
                System.out.println("Number of folds: " + folds);
                int reps = 1;
                double sumAccuracy = 0.0;
                double elapsedTime = 0.0;
                for (int j = 0; j < reps; j++) {
                    data.shuffle(rand);
                    for (int i = 0; i < folds; i++) {
                        int begin = i * data.rows() / folds;
                        int end = (i + 1) * data.rows() / folds;
                        Matrix trainFeatures = new Matrix(data, 0, 0, begin, data.cols() - 1);
                        Matrix trainLabels = new Matrix(data, 0, data.cols() - 1, begin, 1);
                        Matrix testFeatures = new Matrix(data, begin, 0, end - begin, data.cols() - 1);
                        Matrix testLabels = new Matrix(data, begin, data.cols() - 1, end - begin, 1);
                        trainFeatures.add(data, end, 0, data.rows() - end);
                        trainLabels.add(data, end, data.cols() - 1, data.rows() - end);
                        double startTime = System.currentTimeMillis();
                        learner.train(trainFeatures, trainLabels);
                        elapsedTime += System.currentTimeMillis() - startTime;
                        double accuracy = learner.measureAccuracy(testFeatures, testLabels, null);
                        sumAccuracy += accuracy;
                        System.out.println("Rep=" + j + ", Fold=" + i + ", Accuracy=" + accuracy);
                    }
                }
                elapsedTime /= (reps * folds);
                System.out.println("Average time to train (in seconds): " + elapsedTime / 1000.0);
                System.out.println("Mean accuracy=" + (sumAccuracy / (reps * folds)));
                break;
            }
        }
    }

    private class ArgParser {

        String arff;
        String learner;
        String evaluation;
        String evalExtra;
        boolean verbose;
        boolean normalize;

        public ArgParser(String[] argv) {
            for (int i = 0; i < argv.length; i++) {
                switch (argv[i]) {
                    case "-V":
                        verbose = true;
                        break;
                    case "-N":
                        normalize = true;
                        break;
                    case "-A":
                        arff = argv[++i];
                        break;
                    case "-L":
                        learner = argv[++i];
                        break;
                    case "-E":
                        evaluation = argv[++i];
                        switch (argv[i]) {
                            case "static":
                                //expecting a test set name
                                evalExtra = argv[++i];
                                break;
                            case "random":
                                //expecting a double representing the percentage for testing
                                //Note stratification is NOT done
                                evalExtra = argv[++i];
                                break;
                            case "cross":
                                //expecting the number of folds
                                evalExtra = argv[++i];
                                break;
                            case "training":
                                break;
                            default:
                                throw new IllegalArgumentException("Invalid Evaluation Method: " + argv[i]);
                        }
                        break;
                    default:
                        throw new IllegalArgumentException("Invalid parameter: " + argv[i]);
                }
            }
        }

        public String getARFF() {
            return arff;
        }

        public String getLearner() {
            return learner;
        }

        public String getEvaluation() {
            return evaluation;
        }

        public String getEvalParameter() {
            return evalExtra;
        }

        public boolean getVerbose() {
            return verbose;
        }

        public boolean getNormalize() {
            return normalize;
        }
    }

    public static void main(String[] args) throws Exception {
        Main ml = new Main();
        ml.run(args);
    }

}
