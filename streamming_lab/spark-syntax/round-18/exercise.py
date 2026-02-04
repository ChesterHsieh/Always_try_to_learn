# Goal: Get latest record per user using window, then compare with dropDuplicates.
# Expected output:
# 1) Latest per user:
# +-------+----------+------+
# |user_id|dt        |amount|
# +-------+----------+------+
# |u1     |2024-01-03|15.0  |
# |u2     |2024-01-02|5.0   |
# +-------+----------+------+
# 2) dropDuplicates (one row per user; which dt kept may vary):
# - u1 can be (2024-01-01, 10.0) or (2024-01-03, 15.0)
# - u2 is (2024-01-02, 5.0)
# Note: row order may vary; dropDuplicates keeps an arbitrary row per user.
from pyspark.sql import SparkSession, functions as F, Window


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-18-exercise")
        .getOrCreate()
    )

    data = [
        ("u1", "2024-01-01", 10.0),
        ("u1", "2024-01-03", 15.0),
        ("u2", "2024-01-02", 5.0),
    ]
    df = spark.createDataFrame(data, ["user_id", "dt", "amount"])

    # TODO: latest per user by date
    w = Window.partitionBy(___).orderBy(___)
    latest = df.withColumn("rn", F.row_number().over(w)).filter(___).drop("rn")
    latest.show()

    # TODO: dropDuplicates alternative
    dedup = df.dropDuplicates(___)
    dedup.show()

    spark.stop()


if __name__ == "__main__":
    main()
