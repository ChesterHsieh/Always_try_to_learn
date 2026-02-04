# Spark Syntax Interview Rounds

This folder contains 20 rounds of PySpark practice. Each round has:
- `solution.py`: runnable reference solution
- `exercise.py`: fill-in-the-blank practice

## Rounds and Topics
1. SparkSession, DataFrame create, select/withColumn
2. filter/where, column expressions, alias
3. groupBy/agg, count/sum/avg
4. join (inner/left), broadcast
5. orderBy/limit, distinct/dropDuplicates
6. when/otherwise (case-style logic)
7. window functions (row_number, partitionBy)
8. explode, array/map operations
9. UDF vs built-in functions
10. read/write JSON/CSV
11. repartition vs coalesce
12. cache/persist and materialization
13. join strategies and hints
14. AQE basics
15. partition pruning and predicate pushdown
16. skew handling with salting
17. advanced aggregations (approx_count_distinct, rollup)
18. de-duplication (window vs dropDuplicates)
19. small ETL pipeline (filter, join, aggregate)
20. end-to-end with performance trade-offs

## Keywords by Round
### Round 01
`SparkSession`, `createDataFrame`, `withColumn`, `select`, `col`, `expr`, `alias`

### Round 02
`filter`, `where`, `col`, `alias`, `selectExpr`

### Round 03
`groupBy`, `agg`, `count`, `sum`, `avg`, `withColumnRenamed`

### Round 04
`join`, `broadcast`, `inner`, `left`

### Round 05
`orderBy`, `limit`, `distinct`, `dropDuplicates`

### Round 06
`when`, `otherwise`

### Round 07
`Window`, `row_number`, `partitionBy`, `orderBy`

### Round 08
`explode`, `map_keys`, `array`

### Round 09
`udf`, `lower`, `regexp_replace`, `StringType`

### Round 10
`write`, `read`, `json`, `csv`, `option`, `mode`

### Round 11
`repartition`, `coalesce`, `rdd.getNumPartitions`

### Round 12
`persist`, `StorageLevel`, `count`

### Round 13
`hint`, `broadcast`, `shuffle_hash`, `join`

### Round 14
`spark.conf.set`, `spark.sql.adaptive.enabled`

### Round 15
`partitionBy`, `parquet`, `filter`

### Round 16
`rand`, `cast`, `explode`, `array`, `join`

### Round 17
`approx_count_distinct`, `rollup`, `sum`

### Round 18
`Window`, `row_number`, `orderBy`, `dropDuplicates`

### Round 19
`filter`, `select`, `join`, `groupBy`, `sum`

### Round 20
`cache`, `broadcast`, `filter`, `join`, `groupBy`, `sum`

## Performance Notes Index
- Built-in functions beat Python UDFs
- Filter early, select only needed columns
- Broadcast small dimensions when possible
- Repartition vs coalesce (shuffle cost)
- Cache/persist only when reused
- Partition pruning via partitioned writes
- Salting for skewed keys
