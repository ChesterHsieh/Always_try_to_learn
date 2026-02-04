from pyspark.sql import SparkSession, functions as F, Window


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-18")
        .getOrCreate()
    )

    data = [
        ("u1", "2024-01-01", 10.0),
        ("u1", "2024-01-03", 15.0),
        ("u2", "2024-01-02", 5.0),
    ]
    df = spark.createDataFrame(data, ["user_id", "dt", "amount"])

    w = Window.partitionBy("user_id").orderBy(F.col("dt").desc())
    latest = df.withColumn("rn", F.row_number().over(w)).filter("rn = 1").drop("rn")
    latest.show()

    dedup = df.dropDuplicates(["user_id"])
    dedup.show()

    print("Performance: window gives deterministic latest; dropDuplicates is cheaper.")
    spark.stop()


if __name__ == "__main__":
    main()
