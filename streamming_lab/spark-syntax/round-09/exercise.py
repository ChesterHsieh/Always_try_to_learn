# Goal: Build slugs with built-in functions and with a Python UDF.
# Expected output:
# 1) Built-in slug:
# +-----------+-----------+
# |title      |slug_builtin|
# +-----------+-----------+
# |Hello World|hello-world|
# |Spark SQL  |spark-sql  |
# +-----------+-----------+
# 2) UDF slug:
# +-----------+--------+
# |title      |slug_udf|
# +-----------+--------+
# |Hello World|hello-world|
# |Spark SQL  |spark-sql  |
# +-----------+--------+
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType


def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "-")


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-09-exercise")
        .getOrCreate()
    )

    df = spark.createDataFrame([("Hello World",), ("Spark SQL",)], ["title"])

    # TODO: built-in slug
    builtin = df.withColumn(
        "slug_builtin",
        ___,
    )
    builtin.show()

    # TODO: UDF slug
    udf_slug = F.udf(slugify, StringType())
    with_udf = df.withColumn("slug_udf", ___)
    with_udf.show()

    spark.stop()


if __name__ == "__main__":
    main()
