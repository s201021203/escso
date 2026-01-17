// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;
import java.util.Comparator;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;

public class Population {
    private TupleList tl;
    private TestCaseOperation tco;
    private ArrayList<TestCase> myPopulation = new ArrayList<>();

    public Population(TupleList tl, TestCaseOperation tco) {
        this.tl = tl;
        this.tco = tco;
    }

    public int size() {
        return this.myPopulation.size();
    }

    public void addNewPopulation(TestCase tc) {
        this.myPopulation.add(tc);
    }

    public void addNewPopulation(int[] tc) {
        TestCase tempTC = new TestCase(tc, this.tco.convertTCToPoint(tc), this.tl.calculateWeight(tc));
        this.addNewPopulation(tempTC);
    }

    // ✅ FIX: Use Comparator instead of invalid "new 1(this)"
    public void sortAscending() {
        Collections.sort(this.myPopulation, Comparator.comparingInt(tc -> tc.fitness));
    }

    // ✅ FIX: Use Comparator for descending order
    public void sortDescending() {
        Collections.sort(this.myPopulation, (tc1, tc2) -> Integer.compare(tc2.fitness, tc1.fitness));
    }

    public TestCase getPopulationByIndex(int index) {
        return this.myPopulation.get(index);
    }

    public TestCase getLowestFitnessPopulation() {
        return Collections.min(this.myPopulation, Comparator.comparingInt(tc -> tc.fitness));
    }

    public TestCase getHighestFitnessPopulation() {
        return Collections.max(this.myPopulation, Comparator.comparingInt(tc -> tc.fitness));
    }

    public int getPopulationSize() {
        return this.myPopulation.size();
    }

    public int getLowestFitness() {
        return this.getLowestFitnessPopulation().fitness;
    }

    public int getHighestFitness() {
        return this.getHighestFitnessPopulation().fitness;
    }

    public Iterator<TestCase> getIterator() {
        return this.myPopulation.iterator();
    }

    public static void main(String[] args) {
    }
}
