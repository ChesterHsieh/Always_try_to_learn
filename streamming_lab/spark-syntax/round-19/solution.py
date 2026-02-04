from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-19")
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

    completed = df_orders.filter(F.col("status") == "completed").select(
        "user_id", "amount"
    )
    joined = completed.join(df_users, "user_id")
    metrics = joined.groupBy("country").agg(F.sum("amount").alias("revenue"))
    metrics.show()

    print("Performance: filter and select before joins to reduce shuffle.")
    spark.stop()


if __name__ == "__main__":
    main()
