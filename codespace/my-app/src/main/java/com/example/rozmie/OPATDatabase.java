// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;
import java.util.ArrayList;
import java.util.Iterator;

public class OPATDatabase {
   private int[] value;
   private int noOfParameter = 0;
   private TupleList tl;
   private ArrayList<MyTestCase> testCase = new ArrayList();
   private ArrayList<MyStrengthSetting> ss = new ArrayList();

   public OPATDatabase(int[] value) {
      this.value = (int[])value.clone();
   }

   public void addSetting(boolean[] parameterInvolve, int strength) {
      if (strength > this.noOfParameter) {
         this.noOfParameter = strength;
      }

      this.ss.add(new MyStrengthSetting(parameterInvolve, strength));
   }

   public void initializeTestSuite() {
      int[] temp = new int[this.noOfParameter];

      for(int i = 0; i < this.noOfParameter; ++i) {
         temp[i] = this.value[i];
      }

      PairGenerator pg = new PairGenerator(temp, this.noOfParameter);

      while(pg.hasMore()) {
         this.testCase.add(new MyTestCase(pg.getNext()));
      }

      this.horizontalExtensionInitialization();
   }

   private int[] appendArray(int[] original, int addValue) {
      int[] temp = new int[original.length + 1];

      int i;
      for(i = 0; i < original.length; ++i) {
         temp[i] = original[i];
      }

      temp[i] = addValue;
      return temp;
   }

   public void deleteTuples(int[] result) {
      this.tl.deleteTuples(result);
      this.testCase.add(new MyTestCase(result));
   }

   public ArrayList<MyTestCase> getCurrentTestSuite() {
      return this.testCase;
   }

   public int getNumberOfParameter() {
      return this.noOfParameter;
   }

   public void assignParameterValue(int[] parameter) {
      if (parameter.length != this.testCase.size()) {
         System.out.println("Size mismatch");
         System.exit(0);
      }

      ArrayList<MyTestCase> tempTestSuite = new ArrayList();

      for(int i = 0; i < this.testCase.size(); ++i) {
         int[] tempTestCase = this.appendArray(((MyTestCase)this.testCase.get(i)).getTestCase(), parameter[i]);
         this.tl.deleteTuples(tempTestCase);
         tempTestSuite.add(new MyTestCase(tempTestCase));
      }

      this.testCase = tempTestSuite;
   }

   public void horizontalExtensionInitialization() {
      ++this.noOfParameter;
      int[] temp = new int[this.noOfParameter];

      for(int i = 0; i < this.noOfParameter; ++i) {
         temp[i] = this.value[i];
      }

      this.tl = new TupleList(temp);
      this.setStrength();
      Iterator<MyTestCase> itr = this.testCase.iterator();

      while(itr.hasNext()) {
         this.tl.deleteTuples(this.appendArray(((MyTestCase)itr.next()).getTestCase(), 0));
      }

   }

   private void setStrength() {
      Iterator<MyStrengthSetting> itr = this.ss.iterator();

      while(itr.hasNext()) {
         this.tl.addSetting((MyStrengthSetting)itr.next());
      }

   }

   public int calculateWeight(int[] parameter) {
      int count = 0;
      if (parameter.length != this.testCase.size()) {
         System.out.println("Size mismatch");
         System.exit(0);
      }

      for(int i = 0; i < this.testCase.size(); ++i) {
         count += this.tl.calculateWeight(this.appendArray(((MyTestCase)this.testCase.get(i)).getTestCase(), parameter[i]));
      }

      return count;
   }

   public int getCurrentTestSuiteSize() {
      return this.testCase.size();
   }

   public boolean allTuplesCovered() {
      return this.tl.allTuplesCovered();
   }

   public static void main(String[] args) {
      int[] testValue = new int[]{3, 2, 2, 2, 2, 2};
      OPATDatabase oDB = new OPATDatabase(testValue);
      boolean[] prmInv = new boolean[]{true, true, true, true};
      oDB.addSetting(prmInv, 3);
      oDB.initializeTestSuite();
      ArrayList<MyTestCase> tt = oDB.getCurrentTestSuite();
      Iterator<MyTestCase> itr = tt.iterator();

      while(itr.hasNext()) {
         System.out.print(itr.next());
      }

   }
}
