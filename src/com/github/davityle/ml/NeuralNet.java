package com.github.davityle.ml;

import com.github.davityle.ml.writtenbyprofessor.Matrix;
import com.github.davityle.ml.writtenbyprofessor.SupervisedLearner;

import java.util.Arrays;
import java.util.Iterator;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Supplier;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;
import java.util.stream.Stream;
import java.util.stream.StreamSupport;

public class NeuralNet extends SupervisedLearner {


    private final Random random;
    private Network network;

    public NeuralNet(Random random) {
        this.random = random;
    }

    public int train(Matrix features, Matrix labels, Matrix validationSet, Matrix validationLabels) throws Exception {
        Network best = network;
        int count = 0, notImproved = 0;
        double bestAccuracy = 0;
        do {
            count++;
            epoch(features, labels, network);
            features.shuffle(random, labels);

            double accuracy = measureAccuracy(validationSet, validationLabels, null);
            if (accuracy > bestAccuracy) {
                best = network.clone();
                bestAccuracy = accuracy;
                notImproved = 0;
            } else {
                notImproved++;
            }
        } while (!network.stoppingConditions.apply(bestAccuracy, notImproved));
        network = best;

        return count;
    }

    @Override
    public void train(Matrix features, Matrix labels) throws Exception {
        train(features, labels, features, labels);
    }


    @Override
    public void predict(double[] features, double[] labels) throws Exception {
        labels[0] = deNormalize(forwardPropogate(features, network)).findFirst().getAsDouble();
    }

    private double net(double[] nodeWeights, double[] input) {
        return IntStream.range(0, nodeWeights.length)
                .mapToDouble(i -> i < input.length ? nodeWeights[i] * input[i] : nodeWeights[i])
                .sum();
    }

    private double[] output(double[] input, Layer layer) {
        return Arrays.stream(layer.nodes)
                .peek(node -> node.output = 1d / (1d + Math.exp(-net(node.weights, input))))
                .mapToDouble(node -> node.output)
                .toArray();
    }

    private double[] forwardPropogate(double[] input, Network network) {
        for (Layer layer : network) {
            input = output(input, layer);
        }
        return input;
    }

    private void backPropogate(double[] input, double[] expected, Network network) {
        for (Layer layer : network.reverse()) {
            for (int i : IntStream.range(0, layer.nodeCount).toArray()) {
                Layer.Node node = layer.nodes[i];
                if (layer.next == null) {
                    node.error = (expected[i] - node.output) * (node.output * (1 - node.output));
                } else {
                    node.error = Arrays.stream(layer.next.nodes)
                            .mapToDouble(nextNode -> nextNode.weights[i] * nextNode.error)
                            .sum() * node.output * (1 - node.output);
                }
            }
        }

        network.stream().peek(layer -> Arrays.stream(layer.nodes).forEach(node -> {
            double[] in = layer.prev == null ? input : Arrays.stream(layer.prev.nodes).mapToDouble(n -> n.output).toArray();
            for (int j = 0; j < in.length; j++) {
                node.derivative[j] += node.error * in[j];
            }
            for (int j = in.length; j < node.derivative.length; j++)
                node.derivative[in.length] += node.error * 1.0;
        })).forEach(layer -> Arrays.stream(layer.nodes).forEach(node -> {
            for (int i = 0; i < node.weights.length; i++) {
                double delta = network.learningRate * node.derivative[i] + (node.lastDelta[i] * network.momentum);
                node.weights[i] += delta;
                node.lastDelta[i] = delta;
                node.derivative[i] = 0.0;
            }
        }));
    }

    private void epoch(Matrix features, Matrix labels, Network network) {
        for (int i = 0; i < features.rows(); i++) {
            forwardPropogate(features.row(i), network);
            backPropogate(features.row(i), normalize(labels.row(i)).toArray(), network);
        }
    }

    private DoubleStream normalize(double[] data) {
        return Arrays.stream(data).map(d -> (d - network.min) / (network.max - network.min));
    }

    private DoubleStream deNormalize(double[] data) {
        return Arrays.stream(data).map(d -> Math.round((network.max - network.min) * d + network.min));
    }

    public void setNetwork(Network network) {
        this.network = network;
    }

    public static class Network implements Iterable<Layer> {
        public Layer first, last;
        public int depth, nodeCount;
        public double learningRate = .1;
        public double momentum = 0;
        public double max, min;
        public BiFunction<Double, Integer, Boolean> stoppingConditions;

        public Network() {
        }

        public Network(Network network) {
            this.first = network.first;
            this.last = network.last;
        }

        public void addLayer(Layer layer) {

            if (first == null) {
                first = layer;
                last = layer;
            } else {
                last.next = layer;
                layer.prev = last;
                last = layer;
            }
            nodeCount += layer.nodeCount;
            depth++;

        }

        @Override
        public Network clone() {
            return clone(this);
        }

        public Network clone(Network initial) {
            return clone(initial, Network::new);
        }

        public Network clone(Network initial, Supplier<Network> supplier) {
            Network network = supplier.get();
            for (Layer layer : this) {
                network.addLayer(layer.clone());
            }
            network.momentum = initial.momentum;
            network.learningRate = initial.learningRate;
            network.min = initial.min;
            network.max = initial.max;
            network.stoppingConditions = initial.stoppingConditions;

            return network;
        }

        public Stream<Layer> stream() {
            return StreamSupport.stream(spliterator(), false);
        }

        @Override
        public Iterator<Layer> iterator() {
            return new Iterator<Layer>() {
                private Layer current = first;

                @Override
                public boolean hasNext() {
                    return current != null;
                }

                @Override
                public Layer next() {
                    Layer next = current;
                    current = current.next;
                    return next;
                }
            };
        }

        public Network reverse() {
            return new ReverseNetwork(this);
        }

        @Override
        public String toString() {
            return last.toString();
        }

        private static class ReverseNetwork extends Network {

            private ReverseNetwork(Network network) {
                this.first = network.first;
                this.last = network.last;
            }

            @Override
            public Network reverse() {
                return new Network(this);
            }

            @Override
            public Iterator<Layer> iterator() {
                return new Iterator<Layer>() {
                    private Layer current = last;

                    @Override
                    public boolean hasNext() {
                        return current != null;
                    }

                    @Override
                    public Layer next() {
                        Layer next = current;
                        current = current.prev;
                        return next;
                    }
                };
            }

        }
    }

    public static class Layer {

        Layer next, prev;

        final int nodeCount;
        final Node[] nodes;

        public Layer(int nodeCount, int weightCount, Random random) {
            this(nodeCount, weightCount, random, Node::new);
        }

        public Layer(int nodeCount, int weightCount, Random random, Function<double[], Node> supplier) {
            this.nodeCount = nodeCount;
            this.nodes = new Node[nodeCount];
            for (int i = 0; i < nodeCount; i++) {
                double[] weights = new double[weightCount];
                for (int j = 0; j < weights.length; j++) {
                    weights[j] = random.nextDouble();
                }
                nodes[i] = supplier.apply(weights);
            }
        }

        public Layer(Node[] nodes) {
            this.nodeCount = nodes.length;
            this.nodes = nodes;
        }

        @Override
        public Layer clone() {
            return new Layer(Arrays.stream(nodes).map(Node::clone).toArray(Node[]::new));
        }

        public static class Node {
            private double[] derivative;
            private double[] lastDelta;
            private double output;

            final double[] weights;
            double error;

            public Node(double[] weights) {
                this(weights, new double[weights.length], new double[weights.length]);
            }

            public Node(double[] weights, double[] derivative, double[] lastDelta) {
                this.weights = weights;
                this.derivative = derivative;
                this.lastDelta = lastDelta;
            }

            @Override
            public Node clone() {
                Node n = new Node(
                        Arrays.copyOf(weights, weights.length),
                        Arrays.copyOf(derivative, derivative.length),
                        Arrays.copyOf(lastDelta, lastDelta.length)
                );
                n.output = output;
                n.error = error;

                return n;
            }
        }
    }
}

