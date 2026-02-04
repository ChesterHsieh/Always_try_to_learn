from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-14")
        .getOrCreate()
    )

    spark.conf.set("spark.sql.adaptive.enabled", "true")
    print("AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled"))

    df = spark.range(0, 1000).repartition(10)
    agg = df.groupBy((F.col("id") % 10).alias("bucket")).count()
    agg.show()

    print("Performance: AQE can coalesce shuffle partitions at runtime.")
    spark.stop()


if __name__ == "__main__":
    main()
