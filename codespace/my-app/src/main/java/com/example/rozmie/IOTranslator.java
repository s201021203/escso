// Source code is decompiled from a .class file using FernFlower decompiler.
package com.example.rozmie;

import java.util.StringTokenizer;

public class IOTranslator {
   private String io;
   private int[][] IOrel;

   public IOTranslator(String io) {
      this.io = io.replaceAll("(\\r|\\n)", "").replaceAll("\\s+", "");
      this.process();
   }

   private void process() {
      StringTokenizer st = new StringTokenizer(this.io, "{");
      this.IOrel = new int[st.countTokens()][];

      for(int i = 0; st.hasMoreTokens(); ++i) {
         StringTokenizer sq = new StringTokenizer(st.nextToken(), "}");
         StringTokenizer lastT = new StringTokenizer(sq.nextToken(), ",");
         int[] temp = new int[lastT.countTokens()];

         for(int c = 0; lastT.hasMoreTokens(); ++c) {
            temp[c] = Integer.parseInt(lastT.nextToken());
         }

         this.IOrel[i] = temp;
      }

   }

   public int[][] getIORelations(int totalIO) {
      int[][] temp = new int[totalIO][];

      for(int i = 0; i < totalIO; ++i) {
         temp[i] = this.IOrel[i];
      }

      return temp;
   }

   public static void main(String[] args) {
      IOTranslator trns = new IOTranslator("{1, 2, 7, 8}, {0, 1, 2, 9}, {4, 5, 7, 8}, {0, 1, 3, 9}, {0,\r\n3, 8}, {6, 7, 8}, {4, 9}, {1, 3, 4}, {0, 2, 6, 7}, {4, 6}, {2, 3, 4, 8},\r\n{2, 3, 5}, {5, 6}, {0, 6, 8}, {8, 9}, {0, 5}, {1, 3, 5, 9}, {1, 6, 7, 9},\r\n{0, 4}, {0, 2, 3}, {1, 3, 6, 9}, {2, 4, 7, 8}, {0, 2, 6, 9}, {0, 1, 7, 8},\r\n{0, 3, 7, 9}, {3, 4, 7, 8}, {1, 5, 7, 9}, {1, 3, 6, 8}, {1, 2, 5}, {3, 4,\r\n5, 7}, {0, 2, 7, 9}, {1, 2, 3}, {1, 2, 6}, {2, 5, 9}, {3, 6, 7}, {1, 2,\r\n4, 7}, {2, 5,8}, {0, 1, 6, 7}, {3, 5, 8}, {0, 1, 2, 8}, {2, 3, 9}, {1, 5,\r\n8}, {1, 3, 5, 7}, {0, 1, 2, 7}, {2, 4, 5, 7}, {1, 4, 5}, {0, 1, 7, 9}, {0,\r\n1, 3, 6}, {1, 4, 8}, {3, 5, 7, 9}, {0, 6, 7, 9}, {2, 6, 7, 9}, {2, 6, 8},\r\n{2, 3, 6}, {1, 3, 7, 9}, {2, 3, 7}, {0, 2, 7, 8}, {0,1, 6, 9}, {1, 3, 7,\r\n8}, {0, 1, 3, 7}, {1,4}, {0,9,3}, {3,7,9}, {0,6,8,4}, {3,5}, {1,2,8,9},\r\n{0,6}, {0,3,7}, {2,4}, {7,8,9}, {3,7,6}, {3,8,9}, {2,5,6,9}, {4,7,9},\r\n{5,8}, {4,6,7,9}, {6, 9}, {6,7}, {3,4,7}, {4,8}, {0,9}, {0,2,6},\r\n{1,4,8,9}, {7,8}, {5,8,9}, {3,6,7,9}, {4,8,9}, {2,4,6,9}, {4,8,9},\r\n{3,5,9}, {0,4,9}, {0,6,8,9}, {4,5,8}, {2,5}, {3,5,6,8}, {2,4,7},\r\n{4,5,6,7}, {5,7,9}, {3,5,8,9}, {2,9}");
      int[][] io = trns.getIORelations(100);

      for(int i = 0; i < 100; ++i) {
         System.out.print(i + ") ");

         for(int j = 0; j < io[i].length; ++j) {
            System.out.print(io[i][j] + " ");
         }

         System.out.println();
      }

   }
}
