# Goal: Use approx_count_distinct and rollup aggregations.
# Expected output:
# 1) Approx distinct users per city:
# +--------+------------+
# |city    |approx_users|
# +--------+------------+
# |Taipei  |2           |
# |Taichung|1           |
# +--------+------------+
# 2) Rollup sum (includes grand total):
# +--------+----------+
# |city    |sum_amount|
# +--------+----------+
# |Taipei  |45.0      |
# |Taichung|5.0       |
# |null    |50.0      |
# +--------+----------+
# Note: row order may vary.
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-17-exercise")
        .getOrCreate()
    )

    data = [
        ("u1", "Taipei", 10.0),
        ("u2", "Taipei", 20.0),
        ("u3", "Taichung", 5.0),
        ("u2", "Taipei", 15.0),
    ]
    df = spark.createDataFrame(data, ["user_id", "city", "amount"])

    # TODO: approx count distinct
    approx = df.groupBy("city").agg(___)
    approx.show()

    # TODO: rollup sum
    rollup_df = df.rollup(___).agg(___)
    rollup_df.show()

    spark.stop()


if __name__ == "__main__":
    main()
