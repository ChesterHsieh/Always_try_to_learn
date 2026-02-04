from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-08")
        .getOrCreate()
    )

    data = [
        ("u1", ["a", "b"], {"k1": 1}),
        ("u2", ["b"], {"k2": 2}),
    ]
    df = spark.createDataFrame(data, ["user_id", "tags", "attrs"])

    exploded = df.select("user_id", F.explode("tags").alias("tag"))
    exploded.show()

    keys = df.select("user_id", F.map_keys("attrs").alias("attr_keys"))
    keys.show()

    print("Performance: avoid explode on huge arrays unless needed.")
    spark.stop()


if __name__ == "__main__":
    main()
