from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-01")
        .getOrCreate()
    )

    data = [
        ("o1", "A", 2, 10.0),
        ("o2", "B", 1, 20.0),
        ("o3", "A", 3, 5.0),
    ]
    df = spark.createDataFrame(data, ["order_id", "item", "qty", "price"])

    df2 = df.withColumn("total", F.col("qty") * F.col("price"))
    result = df2.select("order_id", "item", "total")
    result.show()

    alt = df.select("order_id", "item", F.expr("qty * price").alias("total"))
    alt.show()

    print("Performance: prefer built-in column expressions over UDFs.")
    spark.stop()


if __name__ == "__main__":
    main()
