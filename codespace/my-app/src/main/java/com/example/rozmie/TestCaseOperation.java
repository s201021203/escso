// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

import java.awt.Point;
import java.util.Random;

public class TestCaseOperation {
   private int[] value;
   private int middlePoint;
   private int maxX;
   private int maxY;

   public int getMaxX() {
      return this.maxX;
   }

   public int getMaxY() {
      return this.maxY;
   }

   public static void main(String[] args) {
      TestCaseOperation tCO = new TestCaseOperation(new int[]{2, 2, 2, 2});
      int[] tc = new int[]{1, 2, 1, 1};
      System.out.println(tCO.convertTCToPoint(tc));
   }

   public int[] convertPointToTC(Point p) {
      int x = p.x;
      int y = p.y;
      int[] tc = new int[this.value.length];
      int[] divisor = new int[this.value.length];
      divisor[divisor.length - 1] = this.value[divisor.length - 1];

      int i;
      for(i = divisor.length - 2; i > this.middlePoint - 1; --i) {
         divisor[i] = this.value[i] * divisor[i + 1];
      }

      divisor[this.middlePoint - 1] = this.value[this.middlePoint - 1];
      --i;

      while(i >= 0) {
         divisor[i] = this.value[i] * divisor[i + 1];
         --i;
      }

      for(i = 0; i < this.middlePoint - 1; ++i) {
         tc[i] = x / divisor[i + 1];
         x %= divisor[i + 1];
      }

      for(tc[i++] = x; i < divisor.length - 1; ++i) {
         tc[i] = y / divisor[i + 1];
         y %= divisor[i + 1];
      }

      tc[i++] = y;

      for(i = 0; i < tc.length; ++i) {
         int var10002 = tc[i]++;
      }

      return tc;
   }

   public Point convertTCToPoint(int[] tw) {
      int[] tc = (int[])tw.clone();

      int x;
      for(x = 0; x < tc.length; ++x) {
         int var10002 = tc[x]--;
      }

      x = 0;
      int multiplier = 1;

      int y;
      for(y = this.middlePoint - 1; y >= 0; --y) {
         x += tc[y] * multiplier;
         multiplier *= this.value[y];
      }

      multiplier = 1;
      y = 0;

      for(int i = tc.length - 1; i >= this.middlePoint; --i) {
         y += tc[i] * multiplier;
         multiplier *= this.value[i];
      }

      Point t = new Point(x, y);
      return t;
   }

   public Point getRandomPoint() {
      int[] tc = this.getRandomTC();
      return this.convertTCToPoint(tc);
   }

   public TestCase getRandomTestCase() {
      int[] tc = this.getRandomTC();
      Point p = this.convertTCToPoint(tc);
      TestCase t = new TestCase(tc, p, 0);
      return t;
   }

   public TestCaseOperation(int[] value) {
      this.value = (int[])value.clone();
      this.middlePoint = value.length / 2;
      this.maxX = 1;
      this.maxY = 1;

      int i;
      for(i = 0; i < this.middlePoint; ++i) {
         this.maxX *= value[i];
      }

      while(i < value.length) {
         this.maxY *= value[i];
         ++i;
      }

   }

   public static void printTC(int[] tc) {
      for(int i = 0; i < tc.length; ++i) {
         System.out.print(tc[i] + " ");
      }

      System.out.println();
   }

   public void printTC(Point p) {
      printTC(this.convertPointToTC(p));
   }

   public int[] add(int[] tc, int amount) {
      if (amount < 0) {
         return this.subtract(tc, Math.abs(amount));
      } else {
         int[] result = (int[])tc.clone();
         int carry = amount;

         for(int i = result.length - 1; i >= 0; --i) {
            result[i] += carry % this.value[i];

            for(carry /= this.value[i]; result[i] > this.value[i]; ++carry) {
               result[i] -= this.value[i];
            }

            if (carry == 0) {
               return result;
            }
         }

         return result;
      }
   }

   public int[] subtract(int[] tc, int amount) {
      if (amount == Integer.MIN_VALUE) {
         amount += (int)(Math.random() * 100.0 + 1.0);
      }

      if (amount < 0) {
         return this.add(tc, Math.abs(amount));
      } else {
         int[] result = (int[])tc.clone();
         int carry = amount;

         for(int i = result.length - 1; i >= 0; --i) {
            result[i] -= carry % this.value[i];

            for(carry /= this.value[i]; result[i] < 1; ++carry) {
               result[i] += this.value[i];
            }

            if (carry == 0) {
               return result;
            }
         }

         return result;
      }
   }

   public int distanceBetweenTc(int[] tc1, int[] tc2) {
      int count = tc1[tc1.length - 1] - tc2[tc1.length - 1];
      int multiplier = 1;

      for(int i = tc1.length - 2; i >= 0; --i) {
         multiplier *= this.value[i + 1];
         int result = tc1[i] - tc2[i];
         count += result * multiplier;
      }

      return count;
   }

   public int[] getRandomTC() {
      int[] tc = new int[this.value.length];

      for(int i = 0; i < tc.length; ++i) {
         tc[i] = this.getRandomValue(i);
      }

      return tc;
   }

   public int getRandomValue(int index) {
      Random rand = new Random();
      return rand.nextInt(this.value[index] - 1 + 1) + 1;
   }

   public int[] normalizeTC(int[] tc) {
      int[] temp = (int[])tc.clone();

      for(int i = 0; i < this.value.length; ++i) {
         if (temp[i] > this.value[i]) {
            while(temp[i] > this.value[i]) {
               temp[i] -= this.value[i];
            }
         } else if (temp[i] < 1) {
            while(temp[i] < 1) {
               temp[i] += this.value[i];
            }
         }
      }

      return temp;
   }

   public Point normalizePoint(Point tc) {
      int x = (int)tc.getX();
      int y = (int)tc.getY();

      while(!this.isXNormal(x)) {
         if (x < 0) {
            x *= -1;
         } else {
            x = this.maxX - 1 - (x - (this.maxX - 1));
         }
      }

      while(!this.isYNormal(y)) {
         if (y < 0) {
            y *= -1;
         } else {
            y = this.maxY - 1 - (y - (this.maxY - 1));
         }
      }

      return new Point(x, y);
   }

   private boolean isXNormal(int x) {
      return x >= 0 && x < this.maxX;
   }

   private boolean isYNormal(int y) {
      return y >= 0 && y < this.maxY;
   }
}
