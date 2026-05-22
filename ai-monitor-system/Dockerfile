FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENLINEAGE_JAR_PATH=/opt/openlineage/openlineage-spark_2.12-1.45.0.jar

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-21-jre-headless \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /opt/openlineage \
    && curl -fL "https://repo1.maven.org/maven2/io/openlineage/openlineage-spark_2.12/1.45.0/openlineage-spark_2.12-1.45.0.jar" \
      -o "${OPENLINEAGE_JAR_PATH}"

WORKDIR /app

COPY pipeline /app/pipeline

RUN pip install --no-cache-dir \
    pyspark==3.5.1 \
    opentelemetry-sdk==1.25.0 \
    opentelemetry-exporter-otlp==1.25.0 \
    prometheus_client>=0.20.0 \
    requests>=2.31.0 \
    kubernetes>=29.0.0

CMD ["python", "-m", "pipeline.job"]
