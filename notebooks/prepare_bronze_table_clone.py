# Databricks notebook source
# Participant Info
dbutils.widgets.text("CATALOG_NAME", "")
CATALOG_NAME = dbutils.widgets.get("CATALOG_NAME")

# COMMAND ----------

# Source Catalog
SOURCE_CATALOG = "wiki_data_medaillon"

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.bronze") 

# COMMAND ----------

spark.sql(f"CREATE OR REPLACE TABLE {CATALOG_NAME}.bronze.wiki_sections_raw SHALLOW CLONE {SOURCE_CATALOG}.bronze.wiki_sections_raw")

# COMMAND ----------


