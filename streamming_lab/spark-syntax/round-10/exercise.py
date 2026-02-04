# Goal: Write JSON/CSV, read them back, and show both DataFrames.
# Expected output (same rows shown twice; CSV reads age as string):
# +-------+---+
# |user_id|age|
# +-------+---+
# |u1     |18 |
# |u2     |25 |
# +-------+---+
import os
import shutil

from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-10-exercise")
        .getOrCreate()
    )

    df = spark.createDataFrame(
        [("u1", 18), ("u2", 25)],
        ["user_id", "age"],
    )

    base_dir = os.path.join(os.path.dirname(__file__), "tmp")
    json_path = os.path.join(base_dir, "json_out")
    csv_path = os.path.join(base_dir, "csv_out")
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)

    # TODO: write JSON and CSV
    df.write.mode(___).json(___)
    df.write.mode(___).option("header", True).csv(___)

    # TODO: read back
    read_json = spark.read.json(___)
    read_csv = spark.read.option("header", True).csv(___)
    read_json.show()
    read_csv.show()

    spark.stop()


if __name__ == "__main__":
    main()
