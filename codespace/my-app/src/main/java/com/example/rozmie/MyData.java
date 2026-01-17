// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

public class MyData {
   private int prevVal = 0;
   private int[] value;
   private boolean[] hasValue;
   private int sequence;
   private MyData[] data;
   private boolean endParameter = false;
   private boolean startParameter = false;
   private MyData prevData;
   private int complete = 0;
   private boolean[] completeValue;
   private boolean[] parameterInv;

   public MyData(int seq, int[] val, boolean[] p) {
      this.sequence = seq;
      this.value = val;
      this.data = new MyData[this.value[this.sequence] + 1];
      this.hasValue = new boolean[this.value[this.sequence] + 1];
      this.completeValue = new boolean[this.value[this.sequence] + 1];
      if (seq == val.length - 1) {
         this.endParameter = true;
      }

      this.startParameter = true;
      this.parameterInv = p;
   }

   public MyData(int seq, int[] val, MyData previous, int prevV, boolean[] p) {
      this.sequence = seq;
      this.value = val;
      this.data = new MyData[this.value[this.sequence] + 1];
      this.hasValue = new boolean[this.value[this.sequence] + 1];
      this.completeValue = new boolean[this.value[this.sequence] + 1];
      if (seq == val.length - 1) {
         this.endParameter = true;
      }

      this.prevData = previous;
      this.prevVal = prevV;
      this.parameterInv = p;
   }

   public boolean nextParameterHas(int v) {
      return this.hasValue[v];
   }

   public MyData getNextParameter(int v) {
      return this.data[v];
   }

   public boolean isNextParameterComplete(int v) {
      return this.completeValue[v];
   }

   public void addData(int[] r) {
      int temp = r[this.sequence];
      if (!this.endParameter) {
         if (!this.completeValue[temp]) {
            if (this.hasValue[temp]) {
               this.data[temp].addData(r);
            } else {
               this.hasValue[temp] = true;
               this.data[temp] = new MyData(this.sequence + 1, this.value, this, temp, this.parameterInv);
               this.data[temp].addData(r);
            }
         }
      } else {
         this.hasValue[temp] = true;
         ++this.complete;
         if (this.checkCompletion()) {
            this.data = null;
            this.prevData.deleteData(this.prevVal);
            this.prevData = null;
         }
      }

   }

   public void deleteData(int s) {
      ++this.complete;
      if (this.checkCompletion()) {
         this.data = null;
         if (!this.startParameter) {
            this.prevData.deleteData(this.prevVal);
            this.prevData = null;
         } else {
            this.completeValue[s] = true;
         }
      } else {
         this.data[s] = null;
         this.completeValue[s] = true;
      }

   }

   private boolean checkCompletion() {
      if (!this.parameterInv[this.sequence]) {
         return this.complete == 1;
      } else {
         return this.complete == this.value[this.sequence] + 1;
      }
   }
}
