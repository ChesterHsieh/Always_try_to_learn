from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-02")
        .getOrCreate()
    )

    data = [
        ("o1", "A", 2, 10.0),
        ("o2", "B", 1, 20.0),
        ("o3", "A", 3, 5.0),
    ]
    df = spark.createDataFrame(data, ["order_id", "item", "qty", "price"])
    df2 = df.withColumn("total", F.col("qty") * F.col("price"))

    filtered = df2.where(F.col("total") > 15)
    result = filtered.select(F.col("item").alias("sku"), "total")
    result.show()

    alt = df2.filter("total > 15").selectExpr("item as sku", "total")
    alt.show()

    print("Performance: filter early to reduce shuffle and scan work.")
    spark.stop()


if __name__ == "__main__":
    main()
