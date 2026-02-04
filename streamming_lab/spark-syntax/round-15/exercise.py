# Goal: Write partitioned Parquet and filter a single partition.
# Expected output:
# +----------+-------+------+
# |dt        |user_id|amount|
# +----------+-------+------+
# |2024-01-01|u1     |10.0  |
# |2024-01-01|u2     |20.0  |
# +----------+-------+------+
import os
import shutil

from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-15-exercise")
        .getOrCreate()
    )

    data = [
        ("2024-01-01", "u1", 10.0),
        ("2024-01-01", "u2", 20.0),
        ("2024-01-02", "u3", 5.0),
    ]
    df = spark.createDataFrame(data, ["dt", "user_id", "amount"])

    base_dir = os.path.join(os.path.dirname(__file__), "tmp")
    path = os.path.join(base_dir, "parquet_out")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    # TODO: write partitioned parquet
    df.write.mode(___).partitionBy(___).parquet(___)

    read_df = spark.read.parquet(path)
    filtered = read_df.filter(___)
    filtered.show()

    spark.stop()


if __name__ == "__main__":
    main()
