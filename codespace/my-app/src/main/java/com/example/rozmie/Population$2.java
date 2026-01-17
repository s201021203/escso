// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

import java.util.Comparator;

class Population$2 implements Comparator<TestCase> {
   private final Population population; // Explicit reference to Population class

   Population$2(Population population) {
      this.population = population;
   }

   public int compare(TestCase o1, TestCase o2) {
      return o2.fitness - o1.fitness;
   }
}
