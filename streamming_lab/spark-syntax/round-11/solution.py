from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-11")
        .getOrCreate()
    )

    df = spark.range(0, 100)
    print("Original partitions:", df.rdd.getNumPartitions())

    rep = df.repartition(4)
    print("Repartition to 4:", rep.rdd.getNumPartitions())

    coal = rep.coalesce(2)
    print("Coalesce to 2:", coal.rdd.getNumPartitions())

    print("Performance: repartition shuffles, coalesce avoids shuffle when reducing.")
    spark.stop()


if __name__ == "__main__":
    main()
