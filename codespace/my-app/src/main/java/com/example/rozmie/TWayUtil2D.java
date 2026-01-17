// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class TWayUtil2D {
   public TupleList tupleList;
   public TestCaseOperation testCaseOperation;

   public TWayUtil2D(int[] value) {
      this.tupleList = new TupleList(value);
      this.testCaseOperation = new TestCaseOperation(value);
   }

   public void addSetting(boolean[] parameterInvolve, int strength) {
      this.tupleList.addSetting(parameterInvolve, strength);
   }

   public void addSetting(int[] prmInv) {
      this.tupleList.addSetting(prmInv);
   }

   public void deleteTuples(int[] tc) {
      this.tupleList.deleteTuples(tc);
   }

   public void deleteTuples(TestCase tc) {
      this.tupleList.deleteTuples(tc.getTestCase());
   }

   public int calculateWeight(int[] tc) {
      return this.tupleList.calculateWeight(tc);
   }

   public boolean allTuplesCovered() {
      return this.tupleList.allTuplesCovered();
   }

   public boolean tupleListNotEmpty() {
      return !this.tupleList.allTuplesCovered();
   }

   public TestCase createRandomTestCase() {
      int[] temp = this.testCaseOperation.getRandomTC();
      Point tempP = this.testCaseOperation.convertTCToPoint(temp);
      int fitness = this.tupleList.calculateWeight(temp);
      return new TestCase(temp, tempP, fitness);
   }

   public ArrayList<TestCase> createRandomPopulation(int numberOfPopulation) {
      ArrayList<TestCase> temp = new ArrayList(numberOfPopulation);

      for(int i = 0; i < numberOfPopulation; ++i) {
         temp.add(this.createRandomTestCase());
      }

      return temp;
   }

   public void sortPopulationAscending(ArrayList<TestCase> temp) {
      TestCase.setAscendingOrder();
      Collections.sort(temp);
   }

   public void sortPopulationDescending(ArrayList<TestCase> temp) {
      TestCase.setDescendingOrder();
      Collections.sort(temp);
   }

   public void updateTestCase(TestCase tc, Point newPoint) {
      newPoint = this.testCaseOperation.normalizePoint(newPoint);
      int[] newTC = this.testCaseOperation.convertPointToTC(newPoint);
      tc.fitness = this.tupleList.calculateWeight(newTC);
      tc.setPoint(newPoint);
      tc.setTC(newTC);
   }

   public void updateTestCase(TestCase tc, int x, int y) {
      this.updateTestCase(tc, new Point(x, y));
   }

   public void updateTestCase(TestCase tc, int[] newTC) {
      newTC = this.testCaseOperation.normalizeTC(newTC);
      Point newPoint = this.testCaseOperation.convertTCToPoint(newTC);
      tc.fitness = this.tupleList.calculateWeight(newTC);
      tc.setPoint(newPoint);
      tc.setTC(newTC);
   }

   public static int getRandomNumber(int start, int end) {
      Random rnd = new Random();
      return rnd.nextInt(end - start + 1) + start;
   }

   public static void printTC(int[] tc) {
      TestCaseOperation.printTC(tc);
   }

   public static void main(String[] args) {
   }
}
