// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

import java.text.DecimalFormat;

public class QLearning {
   private double learningRate;
   private double discountFactor;
   private int noOfStates;
   private int noOfActions;
   private double[][] QTable;
   private int currentState = 0;

   public QLearning(int noOfStates, int noOfActions, double learningRate, double discountFactor) {
      this.noOfActions = noOfActions;
      this.noOfStates = noOfStates;
      this.learningRate = learningRate;
      this.discountFactor = discountFactor;
      this.initializeQTable();
   }

   private void initializeQTable() {
      this.QTable = new double[this.noOfStates][];

      for(int i = 0; i < this.noOfStates; ++i) {
         this.QTable[i] = new double[this.noOfActions];

         for(int j = 0; j < this.noOfActions; ++j) {
            this.QTable[i][j] = 10.0;
         }
      }

   }

   public void updateQValue(double currentReward) {
      int nextState = this.currentState + 1;
      if (nextState >= this.noOfStates) {
         nextState = 0;
      }

      double nextStateBestQ = this.QTable[nextState][this.findMax(this.QTable[nextState])];
      int currentBestAction = this.getBestAction();
      double Qsa = this.QTable[this.currentState][currentBestAction];
      this.QTable[this.currentState][currentBestAction] = Qsa + this.learningRate * (currentReward + this.discountFactor * nextStateBestQ - Qsa);
   }

   public void setCurrentState(int state) {
      this.currentState = state;
   }

   public double getLearningRate() {
      return this.learningRate;
   }

   public void setLearningRate(double learningRate) {
      this.learningRate = learningRate;
   }

   public int getBestAction() {
      return this.findMax(this.QTable[this.currentState]);
   }

   private int findMax(double[] ds) {
      double maxValue = ds[0];
      int maxIndex = 0;

      for(int i = 1; i < ds.length; ++i) {
         if (maxValue < ds[i]) {
            maxValue = ds[i];
            maxIndex = i;
         }
      }

      return maxIndex;
   }

   public void printQTable() {
      DecimalFormat df = new DecimalFormat("0.0");

      for(int i = 0; i < this.noOfStates; ++i) {
         for(int j = 0; j < this.noOfActions; ++j) {
            System.out.print(df.format(this.QTable[i][j]) + " ");
         }

         System.out.println();
      }

   }

   public void manuallySetQValue(double value, int states, int action) {
      this.QTable[states][action] = value;
   }

   public static void main(String[] args) {
      QLearning ql = new QLearning(3, 3, 0.9, 0.1);

      for(int i = 0; i < 100; ++i) {
         ql.printQTable();
         int bestAction = ql.getBestAction();
         System.out.println("Best Action: " + bestAction);
         ql.updateQValue((double)(Math.random() < 0.5 ? -1 : 1));
         ql.setCurrentState(bestAction);
      }

   }
}
