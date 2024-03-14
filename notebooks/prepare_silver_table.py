# Databricks notebook source
# MAGIC %pip install mwclient
# MAGIC %pip install mwparserfromhell
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# Participant Info
CATALOG_NAME = "wiki_data_medaillon"

# COMMAND ----------

# imports
import mwclient  # for downloading example Wikipedia articles
import mwparserfromhell  # for splitting Wikipedia articles into sections
import openai  # for generating embeddings
import pandas as pd  # for DataFrames to store article sections and embeddings
import re  # for cutting <ref> links out of Wikipedia articles
import tiktoken  # for counting tokens
import markdown
from IPython.display import display, Markdown, HTML

# COMMAND ----------

from pyspark.sql.functions import udf, col
df_spark = spark.table(f"{CATALOG_NAME}.bronze.wiki_sections_raw")

# COMMAND ----------

from pyspark.sql.types import StringType, BooleanType

def clean_section(section):
    section = re.sub(r"<ref.*?</ref>", "", section)
    section = section.strip()
    return section

def keep_section(section):
    if len(section) < 16:
        return False
    else:
        return True

clean_section_udf = udf(clean_section, StringType())
keep_section_udf = udf(keep_section, BooleanType())

    

# COMMAND ----------



df = df_spark.withColumn("section_body_cleaned", clean_section_udf(col("section_body")))
df = df.withColumn("keep_section", keep_section_udf(col("section_body")))
df.head(5)

# COMMAND ----------

df = df.where(col("keep_section")).select("section_title", "section_body_cleaned")

# COMMAND ----------

# -> Write into silver table
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG_NAME}.silver")


# Write the spark dataframe to the silver table
#df_spark.write.mode("overwrite").saveAsTable(f"{CATALOG_NAME}.silver.wiki_sections_clean")
df.write.mode("overwrite").saveAsTable(f"{CATALOG_NAME}.silver.wiki_sections_clean")
