// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

import java.util.Comparator;

class Population$1 implements Comparator<TestCase> {
   private final Population population; // Explicit reference to the outer class

   Population$1(Population population) {
      this.population = population;
   }

   public int compare(TestCase o1, TestCase o2) {
      return o1.fitness - o2.fitness;
   }
}
