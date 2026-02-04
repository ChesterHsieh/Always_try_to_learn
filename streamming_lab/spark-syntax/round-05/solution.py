from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-05")
        .getOrCreate()
    )

    data = [
        ("u1", 80),
        ("u2", 90),
        ("u1", 85),
        ("u3", 70),
    ]
    df = spark.createDataFrame(data, ["user_id", "score"])

    top2 = df.orderBy(F.col("score").desc()).limit(2)
    top2.show()

    distinct_users = df.select("user_id").distinct()
    distinct_users.show()

    dedup_users = df.dropDuplicates(["user_id"])
    dedup_users.show()

    print("Performance: limit after orderBy can still sort; use top-N patterns if needed.")
    spark.stop()


if __name__ == "__main__":
    main()
