from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StringType


def slugify(text: str) -> str:
    return text.strip().lower().replace(" ", "-")


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-09")
        .getOrCreate()
    )

    df = spark.createDataFrame([("Hello World",), ("Spark SQL",)], ["title"])

    builtin = df.withColumn(
        "slug_builtin",
        F.lower(F.regexp_replace(F.col("title"), r"\s+", "-")),
    )
    builtin.show()

    udf_slug = F.udf(slugify, StringType())
    with_udf = df.withColumn("slug_udf", udf_slug(F.col("title")))
    with_udf.show()

    print("Performance: built-in functions are faster than Python UDFs.")
    spark.stop()


if __name__ == "__main__":
    main()
