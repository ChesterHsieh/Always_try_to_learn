from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-12")
        .getOrCreate()
    )

    df = spark.range(0, 10000)
    cached = df.persist(StorageLevel.MEMORY_ONLY)

    print("Count:", cached.count())
    print("Sum:", cached.groupBy().sum("id").collect())

    print("Performance: cache/persist only when reused multiple times.")
    spark.stop()


if __name__ == "__main__":
    main()
