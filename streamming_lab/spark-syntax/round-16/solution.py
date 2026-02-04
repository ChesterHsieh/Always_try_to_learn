from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-16")
        .getOrCreate()
    )

    fact = [
        ("k1", 10.0),
        ("k1", 12.0),
        ("k1", 8.0),
        ("k1", 7.0),
        ("k2", 5.0),
    ]
    dim = [("k1", "A"), ("k2", "B")]
    df_fact = spark.createDataFrame(fact, ["key", "amount"])
    df_dim = spark.createDataFrame(dim, ["key", "label"])

    salted_fact = df_fact.withColumn("salt", (F.rand() * 4).cast("int"))
    salts = F.array(F.lit(0), F.lit(1), F.lit(2), F.lit(3))
    salted_dim = df_dim.withColumn("salt", F.explode(salts))

    joined = salted_fact.join(salted_dim, ["key", "salt"])
    joined.show()

    print("Performance: salting spreads skewed keys across partitions.")
    spark.stop()


if __name__ == "__main__":
    main()
