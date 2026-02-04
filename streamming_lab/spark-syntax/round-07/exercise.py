# Goal: Rank users per group and keep top-1 by score.
# Expected output:
# +--------+-------+-----+---+
# |group_id|user_id|score|rn |
# +--------+-------+-----+---+
# |g1      |u2     |15   |1  |
# |g2      |u4     |12   |1  |
# +--------+-------+-----+---+
from pyspark.sql import SparkSession, functions as F, Window


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-07-exercise")
        .getOrCreate()
    )

    data = [
        ("g1", "u1", 10),
        ("g1", "u2", 15),
        ("g2", "u3", 7),
        ("g2", "u4", 12),
    ]
    df = spark.createDataFrame(data, ["group_id", "user_id", "score"])

    # TODO: define window and row_number
    w = Window.partitionBy(___).orderBy(___)
    ranked = df.withColumn("rn", F.row_number().over(w))
    top1 = ranked.where(___)
    top1.show()

    spark.stop()


if __name__ == "__main__":
    main()
