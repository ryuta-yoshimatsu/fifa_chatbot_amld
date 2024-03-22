# Databricks notebook source
# DBTITLE 1,Bronze Table
import dlt 
from pyspark.sql.functions import col, trim, regexp_replace, length

@dlt.create_table()
def bronze_table():
    return spark.readStream.format("cloudFiles").option("cloudFiles.format", "json").load("/Volumes/amld_catalog/raw_data/raw_text_files")

# COMMAND ----------

# DBTITLE 1,Silver Table
@dlt.create_table()
@dlt.expect_or_drop("body length", "length(section_body_cleaned) > 16")
def wiki_sections_clean():

    return (dlt.read("bronze_table")
     .withColumn("section_body_cleaned", trim(regexp_replace(col("section_body"), "<ref.*?</ref>", "")))
     .select("section_title", "section_body_cleaned")
     )
