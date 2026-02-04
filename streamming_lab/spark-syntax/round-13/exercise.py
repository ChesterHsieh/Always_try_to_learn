# Goal: Use join hints to compare broadcast vs shuffle hash join.
# Expected output (two identical tables):
# +---+------+-----+
# |key|amount|label|
# +---+------+-----+
# |1  |100.0 |A    |
# |2  |200.0 |B    |
# +---+------+-----+
from pyspark.sql import SparkSession


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-13-exercise")
        .getOrCreate()
    )

    fact = spark.createDataFrame(
        [(1, 100.0), (2, 200.0), (3, 50.0)],
        ["key", "amount"],
    )
    dim = spark.createDataFrame([(1, "A"), (2, "B")], ["key", "label"])

    # TODO: best: broadcast join
    best = fact.join(dim.hint(___), "key")
    best.show()

    # TODO: secondary: shuffle hash join
    secondary = fact.join(dim.hint(___), "key")
    secondary.show()

    spark.stop()


if __name__ == "__main__":
    main()
