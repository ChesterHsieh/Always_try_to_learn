# Goal: Aggregate by item to get count/sum/avg, then show a sum-only variant.
# Expected output:
# 1) Full aggregation:
# +----+---+----------+----------+
# |item|cnt|sum_amount|avg_amount|
# +----+---+----------+----------+
# |A   |2  |30.0      |15.0      |
# |B   |2  |20.0      |10.0      |
# +----+---+----------+----------+
# 2) Sum-only aggregation:
# +----+----------+
# |item|sum_amount|
# +----+----------+
# |A   |30.0      |
# |B   |20.0      |
# +----+----------+
# Note: row order may vary.
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-03-exercise")
        .getOrCreate()
    )

    data = [
        ("A", 10.0),
        ("A", 20.0),
        ("B", 5.0),
        ("B", 15.0),
    ]
    df = spark.createDataFrame(data, ["item", "amount"])

    # TODO: groupBy item and compute count/sum/avg
    agg_df = df.groupBy("item").agg(
        ___,
        ___,
        ___,
    )
    agg_df.show()

    # TODO: alternative sum-only aggregation
    alt = df.groupBy("item").agg(___).withColumnRenamed(___, ___)
    alt.show()

    spark.stop()


if __name__ == "__main__":
    main()
