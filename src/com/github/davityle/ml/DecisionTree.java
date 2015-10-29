package com.github.davityle.ml;

import com.github.davityle.ml.writtenbyprofessor.Matrix;
import com.github.davityle.ml.writtenbyprofessor.SupervisedLearner;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class DecisionTree extends SupervisedLearner {

    private Node rootNode;

    @Override
    public void train(Matrix features, Matrix labels) throws Exception {
        rootNode = createNode(features, labels, new ArrayList<>());
        System.out.println(rootNode);
    }

    @Override
    public void predict(double[] features, double[] labels) throws Exception {
        labels[0] = rootNode.getAnswer(features);
    }

    private Node createNode(Matrix features, Matrix labels, List<Integer> ints) {
        Node node = nextNode(features, labels, ints);
        if (node.featureIndex > -1 && ints.size() != features.cols()) {
            List<Integer> subInts = new ArrayList<>(ints);
            subInts.add(node.featureIndex);
            node.setSubNodes(IntStream.range(0, features.valueCount(node.featureIndex)).mapToObj(i -> {
                if (node.entropy[i] == 1.0 || node.entropy[i] == 0.0) {
                    return new Node();
                } else {
                    Matrix subFeatures = new Matrix(features, 0, 0, 0, features.cols());
                    Matrix subLabels = new Matrix(labels, 0, 0, 0, labels.cols());
                    for (int j = 0; j < features.rows(); j++) {
                        if (features.get(j, node.featureIndex) == i) {
                            try {
                                subFeatures.add(features, j, 0, 1);
                                subLabels.add(labels, j, 0, 1);
                            } catch (Exception e) {
                                e.printStackTrace();
                            }
                        }
                    }
                    return createNode(subFeatures, subLabels, subInts);
                }
            }).collect(Collectors.toList()));
        }
        return node;
    }

    private Node nextNode(Matrix features, Matrix labels, List<Integer> ints) {
        return IntStream.range(0, features.cols())
                .filter(i -> !ints.contains(i))
                .mapToObj(i -> possibleNode(features, labels, i))
                .min((n1, n2) -> Double.valueOf(n1.gain).compareTo(n2.gain))
                .get();
    }

    private Node possibleNode(Matrix features, Matrix labels, int featureIndex) {
        double[] subAnswers = new double[features.valueCount(featureIndex)];
        double[] subEntropy = new double[features.valueCount(featureIndex)];
        return new Node(featureIndex, IntStream.range(0, features.valueCount(featureIndex))
                .mapToDouble(i -> {
                    double[] answerPercents = new double[labels.valueCount(0)];
                    int total = (int) IntStream.range(0, features.rows())
                            .filter(j -> features.get(j, featureIndex) == i)
                            .peek(j -> answerPercents[(int) labels.get(j, 0)] += 1)
                            .count();
                    IntStream.range(0, answerPercents.length).forEach(j -> answerPercents[j] /= total);
                    subAnswers[i] = IntStream.range(0, answerPercents.length).boxed().max((e1, e2) -> Double.valueOf(answerPercents[e1]).compareTo(answerPercents[e2])).get();
                    subEntropy[i] = entropy(answerPercents);
                    return ((double) total / features.rows()) * subEntropy[i];
                }).sum(), subAnswers, subEntropy);
    }

    private double entropy(double[] valueTotals) {
        return -Arrays.stream(valueTotals).map(value -> value == 0 ? 0 : value * (Math.log(value) / Math.log(2))).sum();
    }

    private static class Node {
        public int featureIndex = -1;
        public double answer, gain;
        public double[] subAnswers;
        public double[] entropy;
        private Optional<List<Node>> subNodesOpt = Optional.empty();

        private Node() {
        }

        private Node(int featureIndex, double gain, double[] subAnswers, double[] entropy) {
            this.featureIndex = featureIndex;
            this.gain = gain;
            this.subAnswers = subAnswers;
            this.entropy = entropy;
        }

        public double getAnswer(double[] val) {
            return subNodesOpt.map(sNodes -> (int) val[featureIndex] >= sNodes.size() ? answer : sNodes.get((int) val[featureIndex]).getAnswer(val)).orElse(answer);
        }

        @Override
        public String toString() {
            return print("", true);
        }

        private String print(String prefix, boolean isTail) {
            StringBuilder builder = new StringBuilder();
            builder.append(prefix).append(isTail ? "└── " : "├── ").append(featureIndex).append(":").append(answer).append('\n');
            if (subNodesOpt.isPresent()) {
                List<Node> subNodes = subNodesOpt.get();
                for (int i = 0; i < subNodes.size() - 1; i++) {
                    builder.append(subNodes.get(i).print(prefix + (isTail ? "    " : "│   "), false));
                }
                if (subNodes.size() > 0) {
                    builder.append(subNodes.get(subNodes.size() - 1).print(prefix + (isTail ? "    " : "│   "), true));
                }
            }
            return builder.toString();
        }

        public void setSubNodes(List<Node> subNodes) {
            this.subNodesOpt = Optional.of(subNodes);
            IntStream.range(0, subNodes.size()).forEach(i -> subNodes.get(i).answer = subAnswers[i]);
        }
    }
}
