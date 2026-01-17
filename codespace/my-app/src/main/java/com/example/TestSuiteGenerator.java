package com.example;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;
import com.example.rozmie.TWayUtil2D;
import com.example.rozmie.TestCase;

public class TestSuiteGenerator {
    public static double[] c = new double[360];
    public static final int MAX_ITER = 100;
    public static final double S = 20.0;
    public static final int numberOfPopulation = 10;
    public static final int MAX_BATCHES = 1;

    // enhancement toggles / parameters
    public static boolean Levy = true;
    private static double beta = 1.5;
    private static Random rnd = new Random();

    private int[] value;
    private int strength;

    public TestSuiteGenerator(int[] value, int strength) {
        this.value = value;
        this.strength = strength;
        initializeRoulleteWheel();
    }

    // ---------- Public API same as before ----------
    public String generateTestSuite() {
        int batchCount = 0;
        int minTestCaseCount = Integer.MAX_VALUE;
        int minBatchIndex = -1;
        StringBuilder result = new StringBuilder();

        while (batchCount < MAX_BATCHES) {
            result.append("Result Set ").append(batchCount + 1).append(":\n");
            TWayUtil2D tt = new TWayUtil2D(value);
            boolean[] pInvolve = new boolean[value.length];
            for (int i = 0; i < value.length; i++) pInvolve[i] = true;
            tt.addSetting(pInvolve, strength);

            int count = 1;
            while (!tt.allTuplesCovered()) {
                int t = 0;
                ArrayList<TestCase> pop = tt.createRandomPopulation(numberOfPopulation);
                TestCase.setDescendingOrder();
                Collections.sort(pop);

                while (pop.get(0).fitness == 0) {
                    pop = tt.createRandomPopulation(numberOfPopulation);
                    Collections.sort(pop);
                }

                TestCase bestTC = pop.get(0).clone();
                double prevFitness = bestTC.fitness;

                // --- SARSA setup ---
                // Two states (example: low/high), two actions (0 = oscillate, 1 = jump/other)
                double[][] Q = new double[2][2];
                double alpha = 0.1;
                double gamma = 0.9;
                double epsilon = 0.2;

                // SARSA optimization loop (keeps Sand Cat structure)
                while (t < MAX_ITER) {
                    double rg = S - ((S) * t / (MAX_ITER));
                    double r = (new Random()).nextDouble() * rg;
                    Iterator<TestCase> itr = pop.iterator();
                    int bestX = bestTC.getX();
                    int bestY = bestTC.getY();

                    int currentState = 0; // start state (you may compute a better state signal)

                    while (itr.hasNext()) {
                        TestCase temp = itr.next();
                        int currentX = temp.getX();
                        int currentY = temp.getY();

                        // Epsilon-greedy over actions
                        int action;
                        if (Math.random() < epsilon) {
                            action = rnd.nextInt(2);
                        } else {
                            double q0 = Q[currentState][0];
                            double q1 = Q[currentState][1];
                            action = (q0 >= q1) ? 0 : 1;
                        }

                        int newX, newY;
                        double oldFitness = temp.fitness;

                        if (action == 0) {
                            // Oscillate move: keep original Sand Cat "R in [-1,1]" style
                            int theta = getTheta();
                            double randX = Math.abs(((new Random()).nextDouble() * bestX) - currentX);
                            double randY = Math.abs(((new Random()).nextDouble() * bestY) - currentY);
                            // NOTE: keep cos/sin usage consistent; original used cos for X and sin for Y
                            newX = (int) (bestX - (r * randX) * Math.cos(2 * Math.PI * theta));
                            newY = (int) (bestY - (r * randY) * Math.sin(2 * Math.PI * theta));
                        } else {
                            // Jump move (exploratory) - same style as Sand Cat jump region
                            newX = (int) (r * (bestX - (new Random()).nextDouble() * currentX));
                            newY = (int) (r * (bestY - (new Random()).nextDouble() * currentY));
                        }

                        // Update candidate in TWay util (same API as original)
                        tt.updateTestCase(temp, newX, newY);

                        // Compute reward: increase in fitness
                        double newFitness = temp.fitness;
                        double reward = newFitness - oldFitness;

                        // Next state heuristic (simple): high if newFitness > bestTC.fitness/2
                        int nextState = (newFitness > bestTC.fitness / 2.0) ? 1 : 0;
                        // Bound nextState
                        if (nextState >= Q.length) nextState = Q.length - 1;

                        // SARSA: choose nextAction greedily (for update)
                        double na0 = Q[nextState][0];
                        double na1 = Q[nextState][1];
                        int nextAction = (na0 >= na1) ? 0 : 1;

                        // SARSA update
                        Q[currentState][action] += alpha * (reward + gamma * Q[nextState][nextAction] - Q[currentState][action]);

                        currentState = nextState;
                    }

                    // Re-sort population and accept improvement like original
                    Collections.sort(pop);
                    TestCase bestCandidate = pop.get(0);
                    if (bestCandidate.fitness > bestTC.fitness) {
                        tt.updateTestCase(bestTC, bestCandidate.getPoint());
                    } else {
                        // No improvement: optionally inject Lévy-based exploratory candidate
                        if (Levy) {
                            double step = getStep();
                            double scaled = step * r;

                            int currentX = bestCandidate.getX();
                            int currentY = bestCandidate.getY();

                            // Move roughly in random direction scaled by problem max bounds (use TWayUtil2D helpers if available)
                            double dx = scaled * tt.testCaseOperation.getMaxX();
                            double dy = scaled * tt.testCaseOperation.getMaxY();

                            int newX = (int) Math.round(currentX + dx);
                            int newY = (int) Math.round(currentY + dy);

                            Point newPoint = new Point(newX, newY);
                            // normalize within bounds (reuse existing helper if present)
                            newPoint = tt.testCaseOperation.normalizePoint(newPoint);

                            int[] newTC = tt.testCaseOperation.convertPointToTC(newPoint);
                            double newWeight = tt.calculateWeight(newTC);
                            pop.add(new TestCase(newTC, newPoint, newWeight));

                            // Keep population size under control (optional: remove worst)
                            Collections.sort(pop);
                            if (pop.size() > numberOfPopulation) {
                                pop.remove(pop.size() - 1);
                            }
                        }
                    }

                    t++;
                } // end optimization

                // Output and delete tuples (same as original)
                result.append(count++).append(") ").append(getTestCaseAsString(bestTC.getTestCase())).append("\n");
                tt.deleteTuples(bestTC.getTestCase());
            } // while !allTuplesCovered

            if (batchCount + 1 < minTestCaseCount) {
                // note: preserved original tracking semantics
                // but original used count - 1; keep same behaviour
            }

            // track minimum among batches
            int produced = count - 1;
            if (produced < minTestCaseCount) {
                minTestCaseCount = produced;
                minBatchIndex = batchCount + 1;
            }

            batchCount++;
            result.append("\n");
        } // end batches

        result.append("Result Set with the least number of test cases:\n");
        result.append("Result Set ").append(minBatchIndex).append(" with ").append(minTestCaseCount).append(" test cases.\n");
        return result.toString();
    }

    // ---------- Roulette wheel & theta helpers (unchanged) ----------
    public static void initializeRoulleteWheel() {
        int[] degree = new int[360];
        int sum = 0;
        for (int i = 0; i < degree.length; i++) {
            degree[i] = i + 1;
            sum += degree[i];
        }
        double[] p = new double[360];
        for (int i = 0; i < degree.length; i++) {
            p[i] = (double) degree[i] / sum;
        }
        p[0] = c[0];
        for (int i = 1; i < degree.length; i++) {
            c[i] = p[i] + c[i - 1];
        }
    }

    public static int getTheta() {
        double r = (new Random()).nextDouble();
        for (int i = 0; i < c.length; i++) {
            if (r <= c[i]) return i;
        }
        return 360;
    }

    // ---------- Lévy helpers ----------
    private static double getStep() {
        double u = rnd.nextGaussian() * doSigma();
        double v = rnd.nextGaussian();
        double eps = 1e-12;
        return u / Math.pow(Math.abs(v) + eps, 1.0 / beta);
    }

    private static double doSigma() {
        double term1 = Math.exp(logGamma(beta + 1.0)) * Math.sin((Math.PI * beta) / 2.0);
        double term2 = Math.exp(logGamma((beta + 1.0) / 2.0)) * beta * Math.pow(2.0, (beta - 1.0) / 2.0);
        return Math.pow((term1 / term2), (1.0 / beta));
    }

    private static double logGamma(double x) {
        double tmp = (x - 0.5) * Math.log(x + 4.5) - (x + 4.5);
        double ser = 1.0
                + 76.18009173 / (x + 0)
                - 86.50532033 / (x + 1)
                + 24.01409822 / (x + 2)
                - 1.231739516 / (x + 3)
                + 0.00120858003 / (x + 4)
                - 0.00000536382 / (x + 5);
        double g = Math.exp(tmp + Math.log(ser * Math.sqrt(2 * Math.PI)));
        return Math.log(g + 1e-300);
    }

    // ---------- Utility ----------
    private String getTestCaseAsString(int[] tc) {
        StringBuilder sb = new StringBuilder();
        for (int val : tc) {
            sb.append(val).append(" ");
        }
        return sb.toString().trim();
    }
}
