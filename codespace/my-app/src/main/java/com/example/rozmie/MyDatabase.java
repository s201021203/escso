// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

import java.util.ArrayList;

public class MyDatabase {
   private int[] parameterInv;
   private boolean[] parameterItr;
   private int noOfParameterInv = 0;
   private int[] value;
   private int strength;
   private MyDataHolder tuppList;
   private PairGenerator tt;
   private boolean hasNext = false;
   private int[] nextTupple;
   private int coveredTupples = 0;
   private int expectedTupples = 0;

   public MyDatabase(int[] value, boolean[] parameterInvolve, int strength) {
      this.value = value;
      this.strength = strength;
      this.parameterItr = (boolean[])parameterInvolve.clone();
      this.initializeObject();
   }

   public MyDatabase(int[] value, boolean[] parameterInvolve) {
      int str = 0;

      for(int i = 0; i < parameterInvolve.length; ++i) {
         if (parameterInvolve[i]) {
            ++str;
         }
      }

      this.value = value;
      this.strength = str;
      this.parameterItr = (boolean[])parameterInvolve.clone();
      this.initializeObject();
   }

   private void initializeObject() {
      this.tuppList = new MyDataHolder(this.value, this.parameterItr);

      int i;
      for(i = 0; i < this.parameterItr.length; ++i) {
         if (this.parameterItr[i]) {
            ++this.noOfParameterInv;
         }
      }

      if (this.noOfParameterInv < this.strength) {
         System.exit(0);
      }

      this.parameterInv = new int[this.noOfParameterInv];
      int count = 0;

      for(i = 0; i < this.parameterItr.length; ++i) {
         if (this.parameterItr[i]) {
            this.parameterInv[count] = i;
            ++count;
         }
      }

      int c;
      for(CombinationGenerator cg = new CombinationGenerator(this.noOfParameterInv, this.strength); cg.hasMore(); this.expectedTupples += c) {
         int[] t = cg.getNext();
         c = 1;

         for(i = 0; i < t.length; ++i) {
            c *= this.value[this.parameterInv[t[i]]];
         }
      }

      this.resetPairGenerator();
   }

   public void resetPairGenerator() {
      int[] tempValue = new int[this.noOfParameterInv];

      for(int i = 0; i < this.noOfParameterInv; ++i) {
         tempValue[i] = this.value[this.parameterInv[i]];
      }

      this.tt = new PairGenerator(tempValue, this.strength);
      this.getNext();
   }

   public boolean hasNext() {
      if (this.tuppList.contains(this.nextTupple)) {
         this.hasNext = false;
         this.getNext();
         return this.hasNext;
      } else {
         return this.hasNext;
      }
   }

   public int getStrength() {
      return this.strength;
   }

   public int[] next() {
      int[] temp = (int[])this.nextTupple.clone();
      this.hasNext = false;
      this.getNext();
      return temp;
   }

   private void getNext() {
      int[] temp;
      do {
         if (!this.tt.hasMore()) {
            return;
         }

         temp = new int[this.value.length];
         int[] t = this.tt.getNext();

         for(int i = 0; i < t.length; ++i) {
            temp[this.parameterInv[i]] = t[i];
         }
      } while(this.tuppList.contains(temp));

      this.nextTupple = (int[])temp.clone();
      this.hasNext = true;
   }

   public boolean isSubset(boolean[] parameterInvolve, int str) {
      if (str < this.strength) {
         return false;
      } else {
         for(int i = 0; i < this.noOfParameterInv; ++i) {
            if (!parameterInvolve[this.parameterInv[i]]) {
               return false;
            }
         }

         return true;
      }
   }

   public boolean isSuperSet(boolean[] parameterInvolve, int str) {
      if (str > this.strength) {
         return false;
      } else {
         for(int i = 0; i < parameterInvolve.length; ++i) {
            if (parameterInvolve[i] && !this.parameterItr[i]) {
               return false;
            }
         }

         return true;
      }
   }

   public int[] calculateWeight(int[] currentResult, int currentNeighbour) {
      int[] weight = new int[this.value[currentNeighbour] + 1];
      if (!this.parameterItr[currentNeighbour]) {
         weight[0] = -1;
         return weight;
      } else {
         int noOfParameterInvolve = 0;

         int i;
         for(i = 0; i < currentResult.length; ++i) {
            if (currentResult[i] != 0 && this.parameterItr[i]) {
               ++noOfParameterInvolve;
            }
         }

         if (noOfParameterInvolve < this.strength - 1) {
            weight[0] = -1;
            return weight;
         } else {
            int[] parameterInvolve = new int[noOfParameterInvolve];
            int count = 0;

            for(i = 0; i < currentResult.length; ++i) {
               if (currentResult[i] != 0 && this.parameterItr[i]) {
                  parameterInvolve[count] = i;
                  ++count;
               }
            }

            int var10002;
            for(CombinationGenerator k = new CombinationGenerator(noOfParameterInvolve, this.strength - 1); k.hasMore(); var10002 = weight[weight.length - 1]++) {
               int[] comb = k.getNext();
               int[] tupple = new int[this.value.length];

               for(i = 0; i < comb.length; ++i) {
                  tupple[parameterInvolve[comb[i]]] = currentResult[parameterInvolve[comb[i]]];
               }

               for(i = 0; i < this.value[currentNeighbour]; ++i) {
                  tupple[currentNeighbour] = i + 1;
                  if (!this.tuppList.contains(tupple)) {
                     var10002 = weight[i]++;
                  }
               }
            }

            return weight;
         }
      }
   }

   public int calculateInitialWeight(int[] currentResult) {
      int weight = 0;
      int noOfParameterInvolve = 0;

      int i;
      for(i = 0; i < currentResult.length; ++i) {
         if (currentResult[i] != 0 && this.parameterItr[i]) {
            ++noOfParameterInvolve;
         }
      }

      if (noOfParameterInvolve < this.strength) {
         return -1;
      } else {
         int[] parameterInvolve = new int[noOfParameterInvolve];
         int count = 0;

         for(i = 0; i < currentResult.length; ++i) {
            if (currentResult[i] != 0 && this.parameterItr[i]) {
               parameterInvolve[count] = i;
               ++count;
            }
         }

         CombinationGenerator k = new CombinationGenerator(noOfParameterInvolve, this.strength);

         while(k.hasMore()) {
            int[] comb = k.getNext();
            int[] tupple = new int[this.value.length];

            for(i = 0; i < comb.length; ++i) {
               tupple[parameterInvolve[comb[i]]] = currentResult[parameterInvolve[comb[i]]];
            }

            if (!this.tuppList.contains(tupple)) {
               ++weight;
            }
         }

         return weight;
      }
   }

   public void permanentDeleteUsedTupples(int[] result) {
      CombinationGenerator k = new CombinationGenerator(this.noOfParameterInv, this.strength);

      while(k.hasMore()) {
         int[] comb = k.getNext();
         int[] tempResult = new int[result.length];

         for(int i = 0; i < this.strength; ++i) {
            tempResult[this.parameterInv[comb[i]]] = result[this.parameterInv[comb[i]]];
         }

         if (!this.tuppList.contains(tempResult)) {
            this.tuppList.addTupple(tempResult);
            ++this.coveredTupples;
         }
      }

   }

   public int getNoOfCoveredPairs() {
      return this.coveredTupples;
   }

   public int getNoOfExpectedTupples() {
      return this.expectedTupples;
   }

   public ArrayList<MyTupples> getUncoveredTupples() {
      ArrayList<MyTupples> temp = new ArrayList();
      this.resetPairGenerator();

      while(this.hasNext) {
         int[] t = this.next();
         temp.add(new MyTupples(t));
      }

      return temp;
   }
}
