# Goal: Use when/otherwise to assign grades by score.
# Expected output:
# +-------+-----+-----+
# |user_id|score|grade|
# +-------+-----+-----+
# |u1     |92   |A    |
# |u2     |78   |B    |
# |u3     |65   |C    |
# +-------+-----+-----+
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-06-exercise")
        .getOrCreate()
    )

    data = [("u1", 92), ("u2", 78), ("u3", 65)]
    df = spark.createDataFrame(data, ["user_id", "score"])

    # TODO: build grade column using when/otherwise
    graded = df.withColumn(
        "grade",
        F.when(___, ___)
        .when(___, ___)
        .otherwise(___),
    )
    graded.show()

    spark.stop()


if __name__ == "__main__":
    main()
