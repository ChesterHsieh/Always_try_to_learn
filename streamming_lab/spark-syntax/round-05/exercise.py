# Goal: Get top-2 scores, list distinct users, and drop duplicate users.
# Expected output:
# 1) Top-2 scores (desc):
# +-------+-----+
# |user_id|score|
# +-------+-----+
# |u2     |90   |
# |u1     |85   |
# +-------+-----+
# 2) Distinct users:
# +-------+
# |user_id|
# +-------+
# |u1     |
# |u2     |
# |u3     |
# +-------+
# 3) dropDuplicates by user_id (one row per user; which score kept may vary):
# +-------+-----+
# |user_id|score|
# +-------+-----+
# |u1     |80/85|
# |u2     |90   |
# |u3     |70   |
# +-------+-----+
# Note: row order may vary; dropDuplicates keeps an arbitrary row per user.
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-05-exercise")
        .getOrCreate()
    )

    data = [
        ("u1", 80),
        ("u2", 90),
        ("u1", 85),
        ("u3", 70),
    ]
    df = spark.createDataFrame(data, ["user_id", "score"])

    # TODO: top 2 scores
    top2 = df.orderBy(___).limit(___)
    top2.show()

    # TODO: distinct users
    distinct_users = df.select(___).distinct()
    distinct_users.show()

    # TODO: drop duplicates by user_id
    dedup_users = df.dropDuplicates(___)
    dedup_users.show()

    spark.stop()


if __name__ == "__main__":
    main()
