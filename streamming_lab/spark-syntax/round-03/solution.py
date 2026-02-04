from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-03")
        .getOrCreate()
    )

    data = [
        ("A", 10.0),
        ("A", 20.0),
        ("B", 5.0),
        ("B", 15.0),
    ]
    df = spark.createDataFrame(data, ["item", "amount"])

    agg_df = df.groupBy("item").agg(
        F.count("*").alias("cnt"),
        F.sum("amount").alias("sum_amount"),
        F.avg("amount").alias("avg_amount"),
    )
    agg_df.show()

    alt = df.groupBy("item").agg({"amount": "sum"}).withColumnRenamed(
        "sum(amount)", "sum_amount"
    )
    alt.show()

    print("Performance: aggregate only needed columns to reduce shuffle size.")
    spark.stop()


if __name__ == "__main__":
    main()
