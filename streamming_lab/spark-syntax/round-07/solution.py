from pyspark.sql import SparkSession, functions as F, Window


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-07")
        .getOrCreate()
    )

    data = [
        ("g1", "u1", 10),
        ("g1", "u2", 15),
        ("g2", "u3", 7),
        ("g2", "u4", 12),
    ]
    df = spark.createDataFrame(data, ["group_id", "user_id", "score"])

    w = Window.partitionBy("group_id").orderBy(F.col("score").desc())
    ranked = df.withColumn("rn", F.row_number().over(w))
    top1 = ranked.where(F.col("rn") == 1)
    top1.show()

    print("Performance: window functions can be expensive; filter columns early.")
    spark.stop()


if __name__ == "__main__":
    main()
