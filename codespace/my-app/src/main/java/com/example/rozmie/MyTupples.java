// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

public class MyTupples {
   private int[] tupples;

   public MyTupples(int[] value) {
      this.tupples = (int[])value.clone();
   }

   public boolean equals(Object obj) {
      MyTupples obj1 = (MyTupples)obj;
      int[] tempTupples = obj1.getTupples();

      for(int i = 0; i < tempTupples.length; ++i) {
         if (tempTupples[i] != this.tupples[i]) {
            return false;
         }
      }

      return true;
   }

   public int[] getTupples() {
      return this.tupples;
   }

   public String toString() {
      String temp = new String();

      for(int i = 0; i < this.tupples.length; ++i) {
         temp = temp + this.tupples[i] + " ";
      }

      temp = temp + "\n";
      return temp;
   }
}
