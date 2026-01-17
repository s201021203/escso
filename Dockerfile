# ---------- Build stage ----------
FROM eclipse-temurin:21-jdk AS build
WORKDIR /src

# Install Maven
RUN apt-get update && \
    apt-get install -y --no-install-recommends maven && \
    rm -rf /var/lib/apt/lists/*

# Copy Maven project (flattened into /src)
COPY codespace/my-app/pom.xml ./pom.xml
COPY codespace/my-app/src ./src

# ✅ Run Maven WHERE pom.xml ACTUALLY IS
RUN mvn -DskipTests clean package

# ---------- Runtime stage ----------
FROM eclipse-temurin:21-jre
WORKDIR /app

# Copy shaded jar
COPY --from=build /src/target/*-shaded.jar app.jar

EXPOSE 8080
CMD ["java", "-jar", "app.jar"]
