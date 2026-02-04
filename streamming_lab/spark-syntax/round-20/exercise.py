# Goal: Filter/ cache fact table, then compare broadcast vs non-broadcast joins.
# Expected output (two identical tables):
# +-----+----------+
# |label|sum_amount|
# +-----+----------+
# |A    |130.0     |
# |B    |50.0      |
# +-----+----------+
# Note: row order may vary.
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-20-exercise")
        .getOrCreate()
    )

    facts = [
        (1, 100.0),
        (1, 30.0),
        (2, 50.0),
        (3, 5.0),
    ]
    dim = [(1, "A"), (2, "B")]
    df_fact = spark.createDataFrame(facts, ["key", "amount"])
    df_dim = spark.createDataFrame(dim, ["key", "label"])

    # TODO: filter, select, and cache
    filtered = df_fact.filter(___).select(___).cache()

    # TODO: best: broadcast join
    best = filtered.join(___, "key")
    best_metrics = best.groupBy("label").agg(F.sum("amount").alias("sum_amount"))
    best_metrics.show()

    # TODO: secondary join without broadcast
    secondary = filtered.join(___, "key")
    secondary_metrics = secondary.groupBy("label").agg(F.sum("amount").alias("sum_amount"))
    secondary_metrics.show()

    spark.stop()


if __name__ == "__main__":
    main()
