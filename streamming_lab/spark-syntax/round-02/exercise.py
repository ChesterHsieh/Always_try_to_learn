# Goal: Filter rows where total > 15 and project item/total with alias.
# Expected output (two identical tables):
# +---+-----+
# |sku|total|
# +---+-----+
# |A  |20.0 |
# |B  |20.0 |
# +---+-----+
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-02-exercise")
        .getOrCreate()
    )

    data = [
        ("o1", "A", 2, 10.0),
        ("o2", "B", 1, 20.0),
        ("o3", "A", 3, 5.0),
    ]
    df = spark.createDataFrame(data, ["order_id", "item", "qty", "price"])
    df2 = df.withColumn("total", F.col("qty") * F.col("price"))

    # TODO: filter rows with total > 15
    filtered = df2.where(___)
    result = filtered.select(F.col("item").alias("sku"), "total")
    result.show()

    # TODO: alternative with filter + selectExpr
    alt = df2.filter(___).selectExpr(___)
    alt.show()

    spark.stop()


if __name__ == "__main__":
    main()
