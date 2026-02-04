# Goal: Enable AQE and show bucket counts after a modulo grouping.
# Expected output:
# AQE enabled: true
# +------+-----+
# |bucket|count|
# +------+-----+
# |0     |100  |
# |1     |100  |
# |2     |100  |
# |3     |100  |
# |4     |100  |
# |5     |100  |
# |6     |100  |
# |7     |100  |
# |8     |100  |
# |9     |100  |
# +------+-----+
# Note: row order may vary.
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-14-exercise")
        .getOrCreate()
    )

    # TODO: enable AQE
    spark.conf.set(___, ___)
    print("AQE enabled:", spark.conf.get("spark.sql.adaptive.enabled"))

    df = spark.range(0, 1000).repartition(10)
    agg = df.groupBy((F.col("id") % 10).alias("bucket")).count()
    agg.show()

    spark.stop()


if __name__ == "__main__":
    main()
