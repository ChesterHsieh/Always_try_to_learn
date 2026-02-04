# Goal: Compare repartition and coalesce partition counts.
# Expected output (exact original count depends on local cores):
# Original partitions: <n>
# Repartition to 4: 4
# Coalesce to 2: 2
from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-11-exercise")
        .getOrCreate()
    )

    df = spark.range(0, 100)
    print("Original partitions:", df.rdd.getNumPartitions())

    # TODO: repartition to 4
    rep = df.repartition(___)
    print("Repartition to 4:", rep.rdd.getNumPartitions())

    # TODO: coalesce to 2
    coal = rep.coalesce(___)
    print("Coalesce to 2:", coal.rdd.getNumPartitions())

    spark.stop()


if __name__ == "__main__":
    main()
