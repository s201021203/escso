package com.example;
import static spark.Spark.*;

public class Main {
    public static void main(String[] args) {
        port(8080);

        // Enable CORS
        options("/*", (req, res) -> {
            res.header("Access-Control-Allow-Origin", "*");
            res.header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
            res.header("Access-Control-Allow-Headers", "Content-Type");
            return "OK";
        });

        before((req, res) -> res.header("Access-Control-Allow-Origin", "*"));

        // Default route
        get("/", (req, res) -> "SCSO Test Suite Generator is running!");

        // API endpoint for test suite generation
        get("/generate", (req, res) -> {
            try {
                // Get parameters
                String strengthStr = req.queryParams("strength");
                String numInputsStr = req.queryParams("numInputs");
                String paramValueStr = req.queryParams("paramValue");

                // Log received parameters
                System.out.println("[DEBUG] Received: strength=" + strengthStr + ", numInputs=" + numInputsStr + ", paramValue=" + paramValueStr);

                // Validate input
                if (strengthStr == null || numInputsStr == null || paramValueStr == null) {
                    res.status(400);
                    return "[ERROR] Missing parameters: strength, numInputs, paramValue";
                }

                // Convert parameters
                int strength = Integer.parseInt(strengthStr);
                int numInputs = Integer.parseInt(numInputsStr);
                int paramValue = Integer.parseInt(paramValueStr);

                // Generate value array based on numInputs and paramValue
                int[] value = new int[paramValue];
                for (int i = 0; i < paramValue; i++) {
                    value[i] = numInputs;
                }

                // Call TestSuiteGenerator
                TestSuiteGenerator generator = new TestSuiteGenerator(value, strength);
                String result = generator.generateTestSuite();

                // Log success
                System.out.println("[SUCCESS] Test suite generated:\n" + result);

                res.type("text/plain");
                return result;

            } catch (Exception e) {
                System.err.println("[ERROR] Exception: " + e.getMessage());
                res.status(500);
                return "[ERROR] Internal Server Error: " + e.getMessage();
            }
        });

        System.out.println("✅ Server started on port 8080...");
    }
}
