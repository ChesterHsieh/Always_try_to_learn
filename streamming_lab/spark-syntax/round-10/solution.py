import os
import shutil

from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-10")
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

    df.write.mode("overwrite").json(json_path)
    df.write.mode("overwrite").option("header", True).csv(csv_path)

    read_json = spark.read.json(json_path)
    read_csv = spark.read.option("header", True).csv(csv_path)
    read_json.show()
    read_csv.show()

    print("Performance: provide schemas to avoid expensive inference.")
    spark.stop()


if __name__ == "__main__":
    main()
