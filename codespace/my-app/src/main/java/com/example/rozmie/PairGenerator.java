// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

public class PairGenerator {
   private int[] value;
   private CombinationGenerator x;
   private int[] pairGenerated;
   private int count = 0;
   private int strength;
   private long[] baseValue;
   private int totalPair;

   public PairGenerator(int[] value, int strength) {
      this.value = (int[])value.clone();
      this.strength = strength;
      this.x = new CombinationGenerator(value.length, strength);
      this.baseValue = new long[strength];
      this.baseValue[strength - 1] = 1L;
      this.generateNewPair();
   }

   private void generateNewPair() {
      if (this.x.hasMore()) {
         this.totalPair = 1;
         this.pairGenerated = this.x.getNext();

         int i;
         for(i = this.strength - 2; i >= 0; --i) {
            this.baseValue[i] = (long)this.value[this.pairGenerated[i + 1]] * this.baseValue[i + 1];
         }

         for(i = 0; i < this.strength; ++i) {
            this.totalPair *= this.value[this.pairGenerated[i]];
         }

         this.count = 0;
      }

   }

   public boolean hasMore() {
      return this.x.hasMore() || this.count != this.totalPair;
   }

   public int[] getNext() {
      int[] result = new int[this.value.length];
      int[] decodedResult = new int[this.strength];
      long tempCount = (long)this.count;

      int i;
      for(i = 0; i < result.length; ++i) {
         result[i] = 0;
      }

      for(i = 0; i < this.strength; ++i) {
         decodedResult[i] = (int)(tempCount / this.baseValue[i]);
         tempCount %= this.baseValue[i];
      }

      for(i = 0; i < this.strength; ++i) {
         result[this.pairGenerated[i]] = decodedResult[i] + 1;
      }

      ++this.count;
      if (this.count == this.totalPair) {
         this.generateNewPair();
      }

      return result;
   }

   public int getTotalPair() {
      return this.totalPair;
   }
}
