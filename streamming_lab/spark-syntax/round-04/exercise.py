# Goal: Join orders with customers using broadcast for inner join and regular left join.
# Expected output:
# 1) Inner join (broadcast):
# +-----------+--------+------+-----+
# |customer_id|order_id|amount|name |
# +-----------+--------+------+-----+
# |c1         |o1      |100.0 |Alice|
# |c2         |o2      |50.0  |Bob  |
# +-----------+--------+------+-----+
# 2) Left join:
# +-----------+--------+------+-----+
# |customer_id|order_id|amount|name |
# +-----------+--------+------+-----+
# |c1         |o1      |100.0 |Alice|
# |c2         |o2      |50.0  |Bob  |
# |c3         |o3      |80.0  |null |
# +-----------+--------+------+-----+
from pyspark.sql import SparkSession, functions as F


def main() -> None:
    spark = (
        SparkSession.builder.master("local[*]")
        .appName("round-04-exercise")
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

    # TODO: inner join with broadcast
    joined = df_orders.join(___, "customer_id", "inner")
    joined.show()

    # TODO: left join without broadcast
    left_join = df_orders.join(___, "customer_id", "left")
    left_join.show()

    spark.stop()


if __name__ == "__main__":
    main()
