
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, isnan, when, count, year, month

# Initialize Spark session
spark = SparkSession.builder.appName("DataWranglingbyPyspark").getOrCreate()

# Define GCS file data path
data_path = "gs://bigdataassignment01/ecommerce_csv/2019-Oct.csv"

# Load CSV file from GCS bucket
df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

# Data cleaning like handling missing values
df_cleaned = df.fillna({"category_code": "unknown_category","brand": "unknown_brand","user_session": "unknown_session"})

# Transform the Data like adding new columns event_year and event_month using event_time column
df_transformed = df_cleaned.withColumn("event_year", year("event_time")).withColumn("event_month", month("event_time"))

# save the output file which is cleaned as CSV format into the GCS bucket
output_path = "gs://bigdataassignment01/ecommerce_csv/cleaned_oct2019_output_csv"
df_transformed.write.option("header", "true").mode("overwrite").csv(output_path)

# print final output
print("Cleaned CSV file saved successfully to:", output_path)

# Stop Spark session
spark.stop()