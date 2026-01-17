package com.example;

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

    // SARSA parameters
    private static final double ALPHA = 0.1;   // learning rate
    private static final double GAMMA = 0.9;   // discount factor
    private static final double epsilon = 0.2; // exploration rate

    private int[] value;
    private int strength;

    public TestSuiteGenerator(int[] value, int strength) {
        this.value = value;
        this.strength = strength;
        initializeRoulleteWheel();
    }

    public String generateTestSuite() {

        StringBuilder testCasesOutput = new StringBuilder();
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

            // SARSA Q-table: [state][action]
            // state: 0 = low fitness, 1 = high fitness
            // action: 0 = oscillate, 1 = jump
            double[][] Q = new double[2][2];

            while (t < MAX_ITER) {

                double rg = S - (S * t / MAX_ITER);
                double r = new Random().nextDouble() * rg;

                int bestX = bestTC.getX();
                int bestY = bestTC.getY();

                Iterator<TestCase> itr = pop.iterator();

                while (itr.hasNext()) {

    TestCase temp = itr.next();
    int currentX = temp.getX();
    int currentY = temp.getY();

    // ✅ SARSA state (environment-based)
    int currentState = (temp.fitness >= bestTC.fitness / 2.0) ? 1 : 0;

    int action;
    if (Math.random() < epsilon) {
        action = new Random().nextInt(2);
    } else {
        double q0 = Q[currentState][0];
        double q1 = Q[currentState][1];
        action = (q0 >= q1) ? 0 : 1;
    }


                    double oldFitness = temp.fitness;
                    int newX, newY;

                    if (action == 0) {
                        // Oscillation (exploration around best)
                        int theta = getTheta();
                        double randX = Math.abs((new Random().nextDouble() * bestX) - currentX);
                        double randY = Math.abs((new Random().nextDouble() * bestY) - currentY);
                        newX = (int) (bestX - (r * randX) * Math.cos(2 * Math.PI * theta));
                        newY = (int) (bestY - (r * randY) * Math.sin(2 * Math.PI * theta));
                    } else {
                        // Jump (exploitation)
                        newX = (int) (r * (bestX - new Random().nextDouble() * currentX));
                        newY = (int) (r * (bestY - new Random().nextDouble() * currentY));
                    }

                    tt.updateTestCase(temp, newX, newY);

                    double newFitness = temp.fitness;
                    double reward = newFitness - oldFitness;

                    int nextState = (newFitness >= bestTC.fitness / 2.0) ? 1 : 0;
                    int nextAction;
                    if (Math.random() < epsilon) {
                        nextAction = new Random().nextInt(2);
                    } else {
                        nextAction = (Q[nextState][0] >= Q[nextState][1]) ? 0 : 1;
                    }

                    // SARSA update
                    Q[currentState][action] +=
                            ALPHA * (reward + GAMMA * Q[nextState][nextAction] - Q[currentState][action]);
                }

                Collections.sort(pop);
                TestCase bestCandidate = pop.get(0);
                if (bestCandidate.fitness > bestTC.fitness) {
                    tt.updateTestCase(bestTC, bestCandidate.getPoint());
                }

                t++;
            }

            testCasesOutput.append(count++)
                    .append(") ")
                    .append(getTestCaseAsString(bestTC.getTestCase()))
                    .append("\n");

            tt.deleteTuples(bestTC.getTestCase());
        }

        StringBuilder result = new StringBuilder();
        result.append(testCasesOutput);

        return result.toString().trim();
    }

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

        c[0] = p[0];
        for (int i = 1; i < degree.length; i++) {
            c[i] = p[i] + c[i - 1];
        }
    }

    public static int getTheta() {
        double r = new Random().nextDouble();
        for (int i = 0; i < c.length; i++) {
            if (r <= c[i]) return i;
        }
        return 359;
    }

    private String getTestCaseAsString(int[] tc) {
        StringBuilder sb = new StringBuilder();
        for (int val : tc) sb.append(val).append(" ");
        return sb.toString().trim();
    }
}