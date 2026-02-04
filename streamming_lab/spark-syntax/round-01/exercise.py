# Goal: Create a total column (qty * price) and show the result using both column
# expressions and expr.
# Expected output (two identical tables):
# +--------+----+-----+
# |order_id|item|total|
# +--------+----+-----+
# |o1      |A   |20.0 |
# |o2      |B   |20.0 |
# |o3      |A   |15.0 |
# +--------+----+-----+
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-01-exercise")
        .getOrCreate()
    )

    data = [
        ("o1", "A", 2, 10.0),
        ("o2", "B", 1, 20.0),
        ("o3", "A", 3, 5.0),
    ]
    df = spark.createDataFrame(data, ["order_id", "item", "qty", "price"])

    # TODO: create total = qty * price
    df2 = df.withColumn("total", ___)
    result = df2.select("order_id", "item", "total")
    result.show()

    # TODO: alternative with expr
    alt = df.select("order_id", "item", ___)
    alt.show()

    spark.stop()


if __name__ == "__main__":
    main()
