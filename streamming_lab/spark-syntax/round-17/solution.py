from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-17")
        .getOrCreate()
    )

    data = [
        ("u1", "Taipei", 10.0),
        ("u2", "Taipei", 20.0),
        ("u3", "Taichung", 5.0),
        ("u2", "Taipei", 15.0),
    ]
    df = spark.createDataFrame(data, ["user_id", "city", "amount"])

    approx = df.groupBy("city").agg(
        F.approx_count_distinct("user_id").alias("approx_users")
    )
    approx.show()

    rollup_df = df.rollup("city").agg(F.sum("amount").alias("sum_amount"))
    rollup_df.show()

    print("Performance: approx_count_distinct trades accuracy for speed.")
    spark.stop()


if __name__ == "__main__":
    main()
