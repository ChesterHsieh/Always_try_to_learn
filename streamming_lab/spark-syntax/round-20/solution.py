from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-20")
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

    filtered = df_fact.filter(F.col("amount") > 10).select("key", "amount").cache()

    best = filtered.join(F.broadcast(df_dim), "key")
    best_metrics = best.groupBy("label").agg(F.sum("amount").alias("sum_amount"))
    best_metrics.show()

    secondary = filtered.join(df_dim, "key")
    secondary_metrics = secondary.groupBy("label").agg(F.sum("amount").alias("sum_amount"))
    secondary_metrics.show()

    print("Performance: filter early + broadcast small dim is best; no broadcast is secondary.")
    spark.stop()


if __name__ == "__main__":
    main()
