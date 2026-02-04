from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-13")
        .getOrCreate()
    )

    fact = spark.createDataFrame(
        [(1, 100.0), (2, 200.0), (3, 50.0)],
        ["key", "amount"],
    )
    dim = spark.createDataFrame([(1, "A"), (2, "B")], ["key", "label"])

    best = fact.join(dim.hint("broadcast"), "key")
    best.show()

    secondary = fact.join(dim.hint("shuffle_hash"), "key")
    secondary.show()

    print("Performance: broadcast is best for small dim; shuffle hash is secondary.")
    spark.stop()


if __name__ == "__main__":
    main()
