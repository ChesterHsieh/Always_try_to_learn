from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-04")
        .getOrCreate()
    )

    orders = [
        ("o1", "c1", 100.0),
        ("o2", "c2", 50.0),
        ("o3", "c3", 80.0),
    ]
    customers = [
        ("c1", "Alice"),
        ("c2", "Bob"),
    ]
    df_orders = spark.createDataFrame(orders, ["order_id", "customer_id", "amount"])
    df_customers = spark.createDataFrame(customers, ["customer_id", "name"])

    joined = df_orders.join(F.broadcast(df_customers), "customer_id", "inner")
    joined.show()

    left_join = df_orders.join(df_customers, "customer_id", "left")
    left_join.show()

    print("Performance: broadcast small dimension tables to avoid shuffle join.")
    spark.stop()


if __name__ == "__main__":
    main()
