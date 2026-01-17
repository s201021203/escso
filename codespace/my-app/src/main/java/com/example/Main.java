package com.example;

import static spark.Spark.*;
import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        /* ===============================
           PORT & HOST (Render compatible)
           =============================== */
        String portEnv = System.getenv("PORT");
        int port = (portEnv != null && !portEnv.isBlank())
                ? Integer.parseInt(portEnv)
                : 8080;

        ipAddress("0.0.0.0");
        port(port);

        /* ===============================
           STATIC FILES (UI)
           =============================== */
        staticFiles.location("/static"); // src/main/resources/static

        get("/", (req, res) -> {
            res.redirect("/main.html");
            return null;
        });

        /* ===============================
           CORS
           =============================== */
        options("/*", (req, res) -> {
            String reqHeaders = req.headers("Access-Control-Request-Headers");
            if (reqHeaders != null) {
                res.header("Access-Control-Allow-Headers", reqHeaders);
            }
            String reqMethod = req.headers("Access-Control-Request-Method");
            if (reqMethod != null) {
                res.header("Access-Control-Allow-Methods", reqMethod);
            }
            res.header("Access-Control-Allow-Origin", "*");
            return "OK";
        });

        before((req, res) -> res.header("Access-Control-Allow-Origin", "*"));

        /* ===============================
           HEALTH CHECK
           =============================== */
        get("/health", (req, res) -> "OK");

        /* ===============================
           GENERATE ENDPOINT (MIXED CONFIG)
           =============================== */
        get("/generate", (req, res) -> {

            String strengthS = req.queryParams("strength");
            String valuesS   = req.queryParams("values");

            System.out.println("[DEBUG] Received strength=" + strengthS + ", values=" + valuesS);

            if (strengthS == null || valuesS == null) {
                res.status(400);
                return "[ERROR] Missing parameters: strength, values";
            }

            try {
                int strength = Integer.parseInt(strengthS);

                String[] parts = valuesS.split(",");
                int[] value = new int[parts.length];

                for (int i = 0; i < parts.length; i++) {
                    value[i] = Integer.parseInt(parts[i].trim());
                    if (value[i] <= 0) {
                        res.status(400);
                        return "[ERROR] All values must be positive integers";
                    }
                }

                if (strength <= 0 || strength > value.length) {
                    res.status(400);
                    return "[ERROR] Invalid strength: must satisfy 1 ≤ t ≤ k";
                }

                System.out.println("[DEBUG] Parsed value[] = " + Arrays.toString(value));

                TestSuiteGenerator generator =
                        new TestSuiteGenerator(value, strength);

                long start = System.currentTimeMillis();
                String result = generator.generateTestSuite();
                long elapsed = System.currentTimeMillis() - start;

                System.out.println("[INFO] Generation completed in " + elapsed + " ms");

                res.type("text/plain");
                return result;

            } catch (NumberFormatException e) {
                res.status(400);
                return "[ERROR] Invalid numeric format: " + e.getMessage();
            } catch (Exception e) {
                e.printStackTrace();
                res.status(500);
                return "[ERROR] Internal Server Error: " + e.getMessage();
            }
        });

        System.out.println("✅ Server started on port " + port);
        System.out.println("✅ UI available at /main.html");
    }
}