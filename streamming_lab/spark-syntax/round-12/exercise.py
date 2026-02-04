# Goal: Persist a DataFrame and reuse it for count and sum.
# Expected output:
# Count: 10000
# Sum: [Row(sum(id)=49995000)]
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-12-exercise")
        .getOrCreate()
    )

    df = spark.range(0, 10000)

    # TODO: persist data
    cached = df.persist(___)

    print("Count:", cached.count())
    print("Sum:", cached.groupBy().sum("id").collect())

    spark.stop()


if __name__ == "__main__":
    main()
