// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

public class MyDataHolder {
   private MyData data;

   public MyDataHolder(int[] value, boolean[] prmtrInv) {
      this.data = new MyData(0, value, prmtrInv);
   }

   public boolean contains(int[] r) {
      if (this.data.isNextParameterComplete(r[0])) {
         return true;
      } else if (!this.data.nextParameterHas(r[0])) {
         return false;
      } else {
         MyData temp = this.data.getNextParameter(r[0]);

         for(int i = 1; i < r.length - 1; ++i) {
            if (temp.isNextParameterComplete(r[i])) {
               return true;
            }

            if (!temp.nextParameterHas(r[i])) {
               return false;
            }

            temp = temp.getNextParameter(r[i]);
         }

         if (temp.isNextParameterComplete(r[r.length - 1])) {
            return true;
         } else {
            return temp.nextParameterHas(r[r.length - 1]);
         }
      }
   }

   public void addTupple(int[] r) {
      if (!this.data.isNextParameterComplete(r[0])) {
         this.data.addData(r);
      }

   }
}
