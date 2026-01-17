// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

import java.awt.Point;

public class TestCase implements Comparable<TestCase> {
   private int[] tc;
   private Point p;
   public int fitness;
   private static boolean descendingOrder = false;

   public TestCase(int[] tc, Point p, int fitness) {
      this.tc = (int[])tc.clone();
      this.p = new Point(p);
      this.fitness = fitness;
   }

   public void setTC(int[] tc) {
      this.tc = (int[])tc.clone();
   }

   public void setPoint(Point p) {
      this.p = (Point)p.clone();
   }

   public boolean equals(Object obj) {
      TestCase temp = (TestCase)obj;
      int[] objTC = temp.getTestCase();

      for(int i = 0; i < this.tc.length; ++i) {
         if (this.tc[i] != objTC[i]) {
            return false;
         }
      }

      return true;
   }

   public int[] getTestCase() {
      return this.tc;
   }

   public Point getPoint() {
      return this.p;
   }

   public int getFitness() {
      return this.fitness;
   }

   public static void setDescendingOrder() {
      descendingOrder = true;
   }

   public static void setAscendingOrder() {
      descendingOrder = false;
   }

   public String toString() {
      String temp = "";

      for(int i = 0; i < this.tc.length; ++i) {
         temp = temp + this.tc[i];
      }

      temp = temp + " => (" + (int)this.p.getX() + ", " + (int)this.p.getY() + ") with fitness = " + this.fitness;
      return temp;
   }

   public static void main(String[] args) {
      int[] value = new int[]{2, 2, 2, 2};
      TestCaseOperation tco = new TestCaseOperation(value);
      Point temp = tco.getRandomPoint();
      TestCase tc = new TestCase(tco.convertPointToTC(temp), temp, 6);
      System.out.println(tc);
   }

   public int compareTo(TestCase o) {
      return descendingOrder ? o.getFitness() - this.fitness : this.fitness - o.getFitness();
   }

   public TestCase clone() {
      return new TestCase((int[])this.tc.clone(), new Point(this.p.x, this.p.y), this.fitness);
   }

   public int getX() {
      return this.p.x;
   }

   public int getY() {
      return this.p.y;
   }
}
