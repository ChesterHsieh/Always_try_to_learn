from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-06")
        .getOrCreate()
    )

    data = [("u1", 92), ("u2", 78), ("u3", 65)]
    df = spark.createDataFrame(data, ["user_id", "score"])

    graded = df.withColumn(
        "grade",
        F.when(F.col("score") >= 90, "A")
        .when(F.col("score") >= 75, "B")
        .otherwise("C"),
    )
    graded.show()

    print("Performance: prefer built-in expressions over Python-side branching.")
    spark.stop()


if __name__ == "__main__":
    main()
