# Goal: Explode tags and extract map keys from attrs.
# Expected output:
# 1) Exploded tags:
# +-------+---+
# |user_id|tag|
# +-------+---+
# |u1     |a  |
# |u1     |b  |
# |u2     |b  |
# +-------+---+
# 2) Map keys:
# +-------+---------+
# |user_id|attr_keys|
# +-------+---------+
# |u1     |[k1]     |
# |u2     |[k2]     |
# +-------+---------+
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-08-exercise")
        .getOrCreate()
    )

    data = [
        ("u1", ["a", "b"], {"k1": 1}),
        ("u2", ["b"], {"k2": 2}),
    ]
    df = spark.createDataFrame(data, ["user_id", "tags", "attrs"])

    # TODO: explode tags
    exploded = df.select("user_id", ___)
    exploded.show()

    # TODO: extract map keys
    keys = df.select("user_id", ___)
    keys.show()

    spark.stop()


if __name__ == "__main__":
    main()
