// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

public class TupleList {
   private int size = 0;
   private int[] value;
   private MyDatabase[] x;
   private int noOfExpectedTuples = 0;

   public TupleList(int[] value) {
      this.value = (int[])value.clone();
   }

   public void addSetting(MyStrengthSetting ss) {
      boolean[] tempPI = ss.getParameterInvolve();
      boolean[] pI = new boolean[this.value.length];

      for(int index = 0; index < this.value.length; ++index) {
         pI[index] = tempPI[index];
      }

      this.addSetting(pI, ss.getStrength());
   }

   public void addSetting(boolean[] parameterInvolve, int strength) {
      if (this.size == 0) {
         this.x = new MyDatabase[1];
         this.x[0] = new MyDatabase(this.value, parameterInvolve, strength);
         this.noOfExpectedTuples += this.x[0].getNoOfExpectedTupples();
         ++this.size;
      } else {
         ++this.size;
         MyDatabase[] temp = new MyDatabase[this.size];

         for(int idx = 0; idx < this.x.length; ++idx) {
            temp[idx] = this.x[idx];
         }

         temp[this.size - 1] = new MyDatabase(this.value, parameterInvolve, strength);
         this.noOfExpectedTuples += temp[this.size - 1].getNoOfExpectedTupples();
         this.x = temp;
      }
   }

   public void addSetting(int[] prmInv) {
      boolean[] parameterInvolve = new boolean[this.value.length];

      for(int idx = 0; idx < parameterInvolve.length; ++idx) {  
         parameterInvolve[idx] = false;
      }

      for(int jdx = 0; jdx < prmInv.length; ++jdx) {  
         parameterInvolve[prmInv[jdx]] = true;
      }

      if (this.size == 0) {
         this.x = new MyDatabase[1];
         this.x[0] = new MyDatabase(this.value, parameterInvolve);
         this.noOfExpectedTuples += this.x[0].getNoOfExpectedTupples();
         ++this.size;
      } else {
         ++this.size;
         MyDatabase[] temp = new MyDatabase[this.size];

         for(int idx = 0; idx < this.x.length; ++idx) {
            temp[idx] = this.x[idx];
         }

         temp[this.size - 1] = new MyDatabase(this.value, parameterInvolve);
         this.noOfExpectedTuples += temp[this.size - 1].getNoOfExpectedTupples();
         this.x = temp;
      }
   }

   public void deleteTuples(int[] result) {
      for(int idx = 0; idx < this.x.length; ++idx) {
         this.x[idx].permanentDeleteUsedTupples(result);
      }
   }

   public int calculateWeight(int[] result) {
      int total = 0;

      for(int idx = 0; idx < this.x.length; ++idx) {
         total += this.x[idx].calculateInitialWeight(result);
      }

      return total;
   }

   public boolean allTuplesCovered() {
      int total = 0;

      for(int idx = 0; idx < this.x.length; ++idx) {
         total += this.x[idx].getNoOfCoveredPairs();
      }

      return this.noOfExpectedTuples == total;
   }
}
