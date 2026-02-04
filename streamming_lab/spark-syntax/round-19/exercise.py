# Goal: Filter completed orders, join users, and sum revenue per country.
# Expected output:
# +-------+-------+
# |country|revenue|
# +-------+-------+
# |TW     |100.0  |
# |US     |80.0   |
# +-------+-------+
# Note: row order may vary.
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-19-exercise")
        .getOrCreate()
    )

    users = [
        ("u1", "TW"),
        ("u2", "TW"),
        ("u3", "US"),
    ]
    orders = [
        ("o1", "u1", 100.0, "completed"),
        ("o2", "u2", 50.0, "pending"),
        ("o3", "u3", 80.0, "completed"),
    ]
    df_users = spark.createDataFrame(users, ["user_id", "country"])
    df_orders = spark.createDataFrame(orders, ["order_id", "user_id", "amount", "status"])

    # TODO: filter completed and select needed columns
    completed = df_orders.filter(___).select(___)
    joined = completed.join(df_users, "user_id")
    metrics = joined.groupBy("country").agg(___)
    metrics.show()

    spark.stop()


if __name__ == "__main__":
    main()
