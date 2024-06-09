# Databricks notebook source
# MAGIC %md
# MAGIC # Predicting Apartment Prices Using Ridge Regression with PySpark
# MAGIC
# MAGIC ### Introduction
# MAGIC
# MAGIC This project aims to predict apartment prices using Ridge regression implemented in PySpark. By leveraging the power of distributed computing, we efficiently process and analyze a large dataset of apartment prices.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Reads a CSV file containing apartment data into a Spark DataFrame using PySpark. It sets options to infer the schema, treat the first row as the header, and use a comma as the delimiter. After loading the data, the DataFrame is registered as a temporary SQL view named "apartments" for easy querying. Finally, the DataFrame is displayed to show its content.
# MAGIC

# COMMAND ----------

file_location = "/FileStore/tables/apartments1-1.csv"
file_type = "csv"


infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)


df.createOrReplaceTempView("apartments")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC Snippet imports necessary libraries and modules for data manipulation and visualization. It includes functions from PySpark for column operations and aggregations, as well as Pandas and Matplotlib for data visualization.
# MAGIC
# MAGIC - `from pyspark.sql.functions import col, sum`: Imports PySpark functions for column operations and summation.
# MAGIC - `import pandas as pd`: Imports Pandas library for data manipulation.
# MAGIC - `import matplotlib.pyplot as plt`: Imports Matplotlib for data plotting.
# MAGIC - `from pyspark.sql import SparkSession`: Imports SparkSession to create a Spark session.
# MAGIC - `import pyspark.sql.functions as F`: Imports additional PySpark functions for various data operations.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, sum
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, regexp_replace
from pyspark.sql.functions import expr

# COMMAND ----------

# MAGIC %md
# MAGIC Calculates and visualizes the missing data counts for each column in the DataFrame.
# MAGIC
# MAGIC - **Calculate Missing Data Counts**: For each column, the number of missing values is computed and converted into a Pandas DataFrame.
# MAGIC - **Print Missing Data Counts**: Displays the DataFrame showing the count of missing values for each column.
# MAGIC - **Visualize Missing Data**: Creates a bar chart to visualize the missing data count for each column, making it easy to identify columns with significant missing data.
# MAGIC

# COMMAND ----------

missing_data_counts = df.select([sum(col(c).isNull().cast("int")).alias(c) for c in df.columns])


# COMMAND ----------

missing_data_counts_pd = missing_data_counts.toPandas().transpose().reset_index()
missing_data_counts_pd.columns = ["column", "missing_count"]
print(missing_data_counts_pd)



plt.figure(figsize=(12, 6))
plt.bar(missing_data_counts_pd["column"], missing_data_counts_pd["missing_count"], color='skyblue')
plt.xlabel('Columns')
plt.ylabel('Missing Data Count')
plt.title('Missing Data Count for Each Column')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Handles duplicates in the DataFrame using SQL.
# MAGIC
# MAGIC 1. **Register DataFrame as SQL View**: Registers the DataFrame as a temporary SQL view named "apartments".
# MAGIC 2. **Count Total Rows**: Uses SQL to count the total number of rows.
# MAGIC 3. **Remove Duplicates**: Uses SQL to select distinct rows, removing duplicates.
# MAGIC 4. **Count Unique and Duplicate Rows**: Calculates the number of unique rows and the number of duplicate rows, then prints these counts.
# MAGIC

# COMMAND ----------

df.createOrReplaceTempView("apartments")
total_count_sql = spark.sql("SELECT COUNT(*) FROM apartments").collect()[0][0]
df_no_duplicates_sql = spark.sql("SELECT DISTINCT * FROM apartments")
unique_count_sql = df_no_duplicates_sql.count()

duplicate_count_sql = total_count_sql - unique_count_sql
print(f"Total rows: {total_count_sql}, Unique rows: {unique_count_sql}, Duplicate rows: {duplicate_count_sql}")

# COMMAND ----------

df_no_duplicates = df.dropDuplicates()

# COMMAND ----------

df_no_address = df_no_duplicates.drop("address")

# COMMAND ----------

# MAGIC %md
# MAGIC Cleans and standardizes the 'price' column.
# MAGIC
# MAGIC 1. **Remove Symbols and Convert to Float**: Removes '$', '֏', and '€' symbols from the 'price' column and converts it to a float.
# MAGIC 2. **Standardize Prices**: Adjusts prices based on the presence of '֏' and '€' symbols, converting '֏' to USD at a rate of 1/400 and '€' to USD at a rate of 1.10.
# MAGIC

# COMMAND ----------

df_cleaned_price = df_no_address.withColumn('price', regexp_replace('price', '[\$,֏,€]', '').cast('float'))
df_cleaned_price = df_cleaned_price.withColumn('price', 
    when(col('price').cast('string').contains('֏'), col('price') / 400)
    .when(col('price').cast('string').contains('€'), col('price') * 1.10)
    .otherwise(col('price'))
)


# COMMAND ----------

# MAGIC %md
# MAGIC Cleans and transforms several columns in the DataFrame by converting categorical values to binary, creating new feature columns, and standardizing data formats for analysis.
# MAGIC

# COMMAND ----------

df_cleaned_price = df_cleaned_price.withColumn('New Construction',
    when(col('New Construction') == 'Yes', 1).otherwise(0)
)

df_cleaned_price = df_cleaned_price.withColumn('Elevator',
    when(col('Elevator') == 'Available', 1).otherwise(0)
)


df_cleaned_price = df_cleaned_price.withColumn('skyscraper',
    when(col('Floors in the Building') <= 5, 0).otherwise(1)
)

df_cleaned_price = df_cleaned_price.withColumn('Ceiling Height',
    regexp_replace(col('Ceiling Height'), '[^0-9]', '').cast('float')
)

df_cleaned_price = df_cleaned_price.withColumn('Ceiling Height',
    when(col('Ceiling Height') <= 2.75, 0).otherwise(1)
)

df_cleaned_price = df_cleaned_price.withColumn('floors_<=6', (col('Floor') <= 6).cast('int'))
df_cleaned_price = df_cleaned_price.withColumn('floors_7_to_13', ((col('Floor') >= 7) & (col('Floor') <= 13)).cast('int'))
df_cleaned_price = df_cleaned_price.withColumn('floors_>13', (col('Floor') > 13).cast('int'))

df_cleaned_price = df_cleaned_price.withColumn('balcony_binary',
    when(col('Balcony').isin('Open balcony', 'Multiple balconies'), 1).otherwise(0)
)

df_cleaned_price = df_cleaned_price.withColumn('Renovation',
    when(col('Renovation').isin('Major Renovation', 'Euro Renovation', 'Designer Renovation', 'Cosmetic Renovation'), 1).otherwise(0)
)

df_cleaned_price = df_cleaned_price.withColumn('furniture_binary',
    when(col('Furniture').isin('Available', 'Partial Furniture'), 1).otherwise(0)
)


# COMMAND ----------

# MAGIC %md
# MAGIC Calculates and visualizes the distribution of provinces in the DataFrame.
# MAGIC
# MAGIC 1. **Count Provinces**: Groups the data by 'province' and counts the occurrences.
# MAGIC 2. **Convert to Pandas**: Converts the result to a Pandas DataFrame.
# MAGIC 3. **Plot Distribution**: Creates a bar chart showing the count of each province.
# MAGIC

# COMMAND ----------

province_counts = df_cleaned_price.groupBy("province").count()
province_counts_pd = province_counts.toPandas()

plt.figure(figsize=(12, 6))
plt.bar(province_counts_pd['province'], province_counts_pd['count'], color='skyblue')
plt.xlabel('Province')
plt.ylabel('Count')
plt.title('Distribution of Provinces')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# COMMAND ----------

rooms_counts = df_cleaned_price.groupBy("number of rooms").count()
rooms_counts_pd = rooms_counts.toPandas()

plt.figure(figsize=(12, 6))
plt.bar(rooms_counts_pd['number of rooms'], rooms_counts_pd['count'], color='skyblue')
plt.xlabel('Number of Rooms')
plt.ylabel('Count')
plt.title('Distribution of Number of Rooms')
plt.tight_layout()
plt.show()


floor_area_pd = df_cleaned_price.select("floor area").toPandas()

plt.figure(figsize=(12, 6))
plt.hist(floor_area_pd['floor area'], bins=50, color='skyblue', edgecolor='black')
plt.xlabel('Floor Area')
plt.ylabel('Frequency')
plt.title('Distribution of Floor Area')
plt.tight_layout()
plt.show()

# COMMAND ----------

df_cleaned_price = df_cleaned_price.withColumn("price", col("price").cast("integer"))
df_cleaned_price = df_cleaned_price.withColumn("Number of Rooms", col("Number of Rooms").cast("integer"))
df_cleaned_price = df_cleaned_price.withColumn("Number of Bathrooms", col("Number of Bathrooms").cast("integer"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Outlier Detection and Removal
# MAGIC
# MAGIC - **Calculate Q1 and Q3:** Determine the first (Q1) and third (Q3) quartiles for the price column to calculate the Interquartile Range (IQR).
# MAGIC - **Define Outlier Bounds:** Establish lower and upper bounds for outliers using the formula:
# MAGIC   - Lower Bound = Q1 - 1.5 * IQR
# MAGIC   - Upper Bound = Q3 + 1.5 * IQR
# MAGIC - **Identify Outliers:** Filter the dataset to find entries with prices outside the defined bounds.
# MAGIC - **Remove Outliers:** Exclude outliers from the dataset to create a new DataFrame without them.
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col, expr

quantiles = df_cleaned_price.approxQuantile("price", [0.25, 0.75], 0.05)
Q1, Q3 = quantiles[0], quantiles[1]
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df_cleaned_price.filter((col("price") < lower_bound) | (col("price") > upper_bound))

df_no_outliers = df_cleaned_price.filter((col("price") >= lower_bound) & (col("price") <= upper_bound))
display(outliers)


# COMMAND ----------

df_pd = df_cleaned_price.select("price").toPandas()
df_no_outliers_pd = df_no_outliers.select("price").toPandas()

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.boxplot(df_pd["price"])
plt.xlabel("Price")
plt.title("Boxplot of Apartment Prices (with Outliers)")

plt.subplot(1, 2, 2)
plt.boxplot(df_no_outliers_pd["price"])
plt.xlabel("Price")
plt.title("Boxplot of Apartment Prices (without Outliers)")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Price Distribution Analysis
# MAGIC
# MAGIC - **Calculate Central Tendencies:** Compute the mean, median, and mode of the apartment prices from the cleaned dataset without outliers.
# MAGIC - **Visualize Price Distribution:** Create a histogram to display the distribution of apartment prices, and add vertical lines to indicate the mean, median, and mode for comparison.
# MAGIC

# COMMAND ----------

mean_price = df_no_outliers.select(F.mean("price")).collect()[0][0]
median_price = df_no_outliers.approxQuantile("price", [0.5], 0.01)[0]
price_pandas = df_no_outliers.select("price").toPandas()
mode_price = price_pandas["price"].mode()[0]


price_series = price_pandas["price"]

plt.figure(figsize=(12, 6))
plt.hist(price_series, bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Price Distribution')
plt.axvline(mean_price, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_price:.2f}')
plt.axvline(median_price, color='green', linestyle='dashed', linewidth=2, label=f'Median: {median_price:.2f}')
plt.axvline(mode_price, color='blue', linestyle='dashed', linewidth=2, label=f'Mode: {mode_price:.2f}')


plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Distribution of Prices')
plt.legend()
plt.tight_layout()
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Short Overview of the ML Workflow
# MAGIC
# MAGIC - **Initialize Spark Session:** Start a Spark session to handle data operations.
# MAGIC - **Load and Clean Data:** Load the dataset, handle missing values, and convert categorical features into numerical indices.
# MAGIC - **Feature Engineering:** Combine feature columns into a single vector, generate polynomial features, and standardize the data.
# MAGIC - **Train Model:** Use linear regression with cross-validation and hyperparameter tuning to train the model.
# MAGIC - **Evaluate Model:** Assess the model's performance using metrics like RMSE (Root Mean Squared Error).
# MAGIC

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler, PolynomialExpansion, StandardScaler

# COMMAND ----------

# MAGIC %md
# MAGIC ### Short Description:
# MAGIC
# MAGIC - **Initialize Spark Session:** Create a Spark session to manage data operations.
# MAGIC - **Handle Missing Values:** Drop rows with null values from the dataset to ensure data quality.
# MAGIC

# COMMAND ----------

spark = SparkSession.builder.appName("ImprovedLinearRegressionExample").getOrCreate()
df_no_outliers = df_no_outliers.dropna()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Index Categorical Columns
# MAGIC
# MAGIC - **Index Categorical Columns:** Convert categorical columns into numerical indices using `StringIndexer` with `handleInvalid` set to "keep" to handle unseen labels.
# MAGIC

# COMMAND ----------

categorical_cols = ['province', 'Construction Type', 'Balcony', 'Furniture']
indexers = [StringIndexer(inputCol=col, outputCol=col + "_indexed", handleInvalid="keep") for col in categorical_cols]
pipeline = Pipeline(stages=indexers)
df_indexed = pipeline.fit(df_no_outliers).transform(df_no_outliers)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Assemble Features
# MAGIC
# MAGIC - **Assemble Features:** Combine all feature columns into a single vector column using `VectorAssembler`, handling invalid entries with `handleInvalid` set to "keep".
# MAGIC

# COMMAND ----------

feature_cols = ['New Construction', 'Elevator', 'Floors in the Building', 'Floor Area', 'Ceiling Height', 'Floor', 
                'Renovation', 'skyscraper', 'floors_<=6', 'floors_7_to_13', 'floors_>13', 'balcony_binary', 
                'furniture_binary','Number of Rooms', 'Number of Bathrooms',] + [col + "_indexed" for col in categorical_cols]

assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep")
df_assembled = assembler.transform(df_indexed)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Ensure Feature Columns are Double
# MAGIC
# MAGIC - **Convert Feature Columns:** Ensure all feature columns are of type `double` for compatibility with machine learning algorithms.
# MAGIC

# COMMAND ----------

for feature in feature_cols:
    df_indexed = df_indexed.withColumn(feature, col(feature).cast('double'))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Create Polynomial Features
# MAGIC
# MAGIC - **Polynomial Features:** Generate polynomial features to capture non-linear relationships using `PolynomialExpansion`.
# MAGIC

# COMMAND ----------

poly_expansion = PolynomialExpansion(degree=2, inputCol="features", outputCol="poly_features")
df_poly = poly_expansion.transform(df_assembled)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Standardize Features
# MAGIC
# MAGIC - **Standardize Features:** Normalize the feature vectors using `StandardScaler` to ensure uniformity across all features.
# MAGIC

# COMMAND ----------

scaler = StandardScaler(inputCol="poly_features", outputCol="scaled_features")
scaler_model = scaler.fit(df_poly)
df_scaled = scaler_model.transform(df_poly)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7: Split the Data
# MAGIC
# MAGIC - **Train-Test Split:** Divide the data into training and test sets using an 80-20 split for model training and evaluation.
# MAGIC

# COMMAND ----------

train_data, test_data = df_scaled.randomSplit([0.8, 0.2], seed=1234)



# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8: Train a Ridge Regression Model
# MAGIC
# MAGIC - **Train Model:** Use a Ridge regression model by setting the `elasticNetParam` to 0.5 in `LinearRegression` to incorporate L2 regularization.
# MAGIC

# COMMAND ----------

lr = LinearRegression(featuresCol="scaled_features", labelCol="price", elasticNetParam=0.5)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 9: Hyperparameter Tuning
# MAGIC
# MAGIC - **Hyperparameter Tuning:** Perform hyperparameter tuning using `ParamGridBuilder` to test different values for `regParam` and use `CrossValidator` with 5-fold cross-validation to find the best model.
# MAGIC

# COMMAND ----------

param_grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1, 1.0]).build()
crossval = CrossValidator(estimator=lr, estimatorParamMaps=param_grid, evaluator=RegressionEvaluator(labelCol="price"), numFolds=5)

cv_model = crossval.fit(train_data)
best_model = cv_model.bestModel



# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 10: Evaluate the Model
# MAGIC
# MAGIC - **Model Evaluation:** Generate predictions on the test data using the best model obtained from cross-validation.
# MAGIC

# COMMAND ----------

predictions = best_model.transform(test_data)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Calculate R², MAE, and RMSE
# MAGIC
# MAGIC - **Model Metrics:** Calculate and print the R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to evaluate the model's performance.
# MAGIC

# COMMAND ----------

evaluator_r2 = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="r2")
evaluator_mae = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="mae")
evaluator_rmse = RegressionEvaluator(labelCol="price", predictionCol="prediction", metricName="rmse")

r2 = evaluator_r2.evaluate(predictions)
mae = evaluator_mae.evaluate(predictions)
rmse = evaluator_rmse.evaluate(predictions)

print(f"R²: {r2}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 11: Visualization
# MAGIC
# MAGIC - **Visualize Predictions:** Convert predictions to a Pandas DataFrame and create a scatter plot to compare actual prices with predicted prices, along with a reference line indicating perfect predictions.
# MAGIC

# COMMAND ----------

predictions_pd = predictions.select("price", "prediction").toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(predictions_pd['price'], predictions_pd['prediction'], alpha=0.5)
plt.plot([predictions_pd['price'].min(), predictions_pd['price'].max()], 
         [predictions_pd['price'].min(), predictions_pd['price'].max()], 
         color='red', lw=2)
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.title('Actual vs Predicted Price')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### An Enhanced Approach to Solving Regression Problems: Increasing R² through Clustering and Feature Engineering
# MAGIC
# MAGIC **Abstract:**
# MAGIC In the quest to improve the predictive accuracy of regression models, particularly the coefficient of determination (R²), this paper introduces an enhanced approach that combines clustering techniques with traditional regression analysis. By first segmenting the dataset into homogeneous groups using clustering algorithms, we can capture underlying patterns and heterogeneity within the data. These clusters are then used as additional features in the regression model, leading to more precise predictions. This method is particularly beneficial when dealing with complex datasets that exhibit significant variability, such as real estate data, where various qualitative and quantitative factors influence prices. The proposed approach demonstrates how preprocessing steps, effective feature engineering, and the integration of clustering can synergistically enhance model performance, offering a robust framework for achieving higher R² values in regression tasks.
# MAGIC ```

# COMMAND ----------



from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.clustering import KMeans
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import col, expr
from pyspark.ml.evaluation import ClusteringEvaluator
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator, ClusteringEvaluator



# COMMAND ----------

description = df.describe()


# COMMAND ----------


spark = SparkSession.builder.appName("DataPreprocessing").getOrCreate()
df = df.dropDuplicates()

province_indexer_n = StringIndexer(inputCol="province", outputCol="province_index")
df = province_indexer_n.fit(df).transform(df)

address_indexer_n = StringIndexer(inputCol="address", outputCol="address_index")
df = address_indexer_n.fit(df).transform(df)


# COMMAND ----------

df = df.withColumn('price', regexp_replace('price', '[\$,֏,€]', '').cast('float'))
df = df.withColumn('price', 
    when(col('price').cast('string').contains('֏'), col('price') / 400)
    .when(col('price').cast('string').contains('€'), col('price') * 1.10)
    .otherwise(col('price'))
)

# COMMAND ----------

df = df.withColumn('New Construction',
    when(col('New Construction') == 'Yes', 1).otherwise(0)
)

df = df.withColumn('Elevator',
    when(col('Elevator') == 'Available', 1).otherwise(0)
)

df = df.withColumn('Ceiling Height',
    regexp_replace(col('Ceiling Height'), '[^0-9]', '').cast('float')
)


# COMMAND ----------



string_columns_to_index = [ "Construction Type", "Balcony", "Furniture", "Renovation"]
string_columns_to_int = ["Number of Rooms", "Number of Bathrooms"]


for column in string_columns_to_index:
    indexer = StringIndexer(inputCol=column, outputCol=column + "_index")
    df = indexer.fit(df).transform(df)

for column in string_columns_to_int:
    df = df.withColumn(column + "_int", df[column].cast("integer"))


columns_to_drop = string_columns_to_index + string_columns_to_int + ["province", "address"]
df = df.drop(*columns_to_drop)


# COMMAND ----------

quantiles = df.approxQuantile("price", [0.25, 0.75], 0.05)
Q1, Q3 = quantiles[0], quantiles[1]
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df.filter((col("price") >= lower_bound) & (col("price") <= upper_bound))


# COMMAND ----------

# MAGIC %md
# MAGIC ### Improving Regression Model Performance Using Clustering Techniques: A Detailed Implementation with K-Means and Linear Regression
# MAGIC
# MAGIC **Abstract:**
# MAGIC This study investigates a method to enhance regression model performance by incorporating clustering techniques. By applying K-Means clustering, we can segment the dataset into distinct groups, which serve as additional features for linear regression models. This comprehensive approach includes data preprocessing, clustering evaluation using Silhouette Score, and regression analysis. Detailed metrics and model summaries are provided to demonstrate the effectiveness of this integrated technique.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clustering Analysis Overview
# MAGIC
# MAGIC ### Steps:
# MAGIC 1. **Initialize Spark Session**:
# MAGIC    - Set up the environment for running Spark applications.
# MAGIC
# MAGIC 2. **Data Preparation**:
# MAGIC    - **Type Casting**: Ensure data types are consistent for all features.
# MAGIC    - **Handle Missing Values**: Fill missing values with zeros to prepare data for analysis.
# MAGIC
# MAGIC 3. **Feature Engineering**:
# MAGIC    - **Vector Assembler**: Combine features into a single vector per instance, required for machine learning models.
# MAGIC
# MAGIC 4. **K-Means Clustering**:
# MAGIC    - Implement K-Means to discover inherent groupings in the data.
# MAGIC    - Determine the optimal number of clusters using:
# MAGIC      - **Elbow Method**: Plotting WSSSE (Within-Sum-of-Squared-Errors) against different k values to find the 'elbow' point.
# MAGIC      - **Silhouette Score**: Assess the clustering quality by calculating the Silhouette Score for different k values.
# MAGIC
# MAGIC ### Visualizations:
# MAGIC - **Elbow Plot**: Helps in identifying the optimal number of clusters by showing where increases in k lead to diminishing returns in WSSSE.
# MAGIC - **Silhouette Plot**: Evaluates how well-separated the resulting clusters are for different numbers of clusters.
# MAGIC
# MAGIC ### Conclusion:
# MAGIC - These analyses and visual tools aid in selecting the best clustering configuration and in understanding the data’s structure for further predictions or insights.
# MAGIC

# COMMAND ----------



spark = SparkSession.builder \
    .appName("ClusteringAnalysis") \
    .getOrCreate()


if not spark._jvm:
    raise Exception("Spark session is not active")


df = df.withColumn("price", col("price").cast("double")) \
       .withColumn("New Construction", col("New Construction").cast("int")) \
       .withColumn("Elevator", col("Elevator").cast("int")) \
       .withColumn("Floors in the Building", col("Floors in the Building").cast("int")) \
       .withColumn("Floor Area", col("Floor Area").cast("int")) \
       .withColumn("Ceiling Height", col("Ceiling Height").cast("float")) \
       .withColumn("Floor", col("Floor").cast("int")) \
       .withColumn("province_index", col("province_index").cast("double")) \
       .withColumn("address_index", col("address_index").cast("double")) \
       .withColumn("Construction Type_index", col("Construction Type_index").cast("double")) \
       .withColumn("Balcony_index", col("Balcony_index").cast("double")) \
       .withColumn("Furniture_index", col("Furniture_index").cast("double")) \
       .withColumn("Renovation_index", col("Renovation_index").cast("double")) \
       .withColumn("Number of Rooms_int", col("Number of Rooms_int").cast("int")) \
       .withColumn("Number of Bathrooms_int", col("Number of Bathrooms_int").cast("int"))


feature_columns = ["price", "New Construction", "Elevator", "Floors in the Building", "Floor Area", 
                   "Ceiling Height", "Floor", "province_index", "address_index", "Construction Type_index", 
                   "Balcony_index", "Furniture_index", "Renovation_index", "Number of Rooms_int", "Number of Bathrooms_int"]
df = df.na.fill(0, subset=feature_columns)


assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
df = assembler.transform(df)


df.select("features").show(5)

wssse_list = []
k_values = range(2, 11)

for k in k_values:
    kmeans = KMeans(k=k, seed=1, featuresCol="features")
    model = kmeans.fit(df)
    wssse = model.summary.trainingCost
    wssse_list.append(wssse)


plt.figure(figsize=(10, 6))
plt.plot(k_values, wssse_list, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("WSSSE")
plt.title("Elbow Method for Optimal k")
plt.show()


silhouette_scores = []

for k in k_values:
    kmeans = KMeans(k=k, seed=1, featuresCol="features", predictionCol="cluster")
    model = kmeans.fit(df)
    clustered_df = model.transform(df)
    evaluator = ClusteringEvaluator(featuresCol='features', predictionCol='cluster', metricName='silhouette')
    silhouette = evaluator.evaluate(clustered_df)
    silhouette_scores.append(silhouette)


plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method for Optimal k")
plt.show()




# COMMAND ----------

# MAGIC %md
# MAGIC ## Clustering Analysis with K-Means in PySpark
# MAGIC
# MAGIC ### Process Overview:
# MAGIC
# MAGIC 1. **Data Splitting**:
# MAGIC    - Split the dataset into 80% training and 20% testing sets using a reproducible seed.
# MAGIC
# MAGIC 2. **Feature Preparation**:
# MAGIC    - Use `VectorAssembler` to combine selected features into a single vector column, required for the K-Means algorithm.
# MAGIC
# MAGIC 3. **Pipeline Setup**:
# MAGIC    - Construct a `Pipeline` comprising the `VectorAssembler` and `K-Means` stages to streamline the workflow.
# MAGIC
# MAGIC 4. **Model Training**:
# MAGIC    - Fit the pipeline on the training data to train the K-Means model with 5 clusters.
# MAGIC
# MAGIC 5. **Data Transformation**:
# MAGIC    - Transform both training and testing datasets using the trained model to assign a cluster to each instance.
# MAGIC
# MAGIC 6. **Results Display**:
# MAGIC    - Show the `price` and `cluster` assignments for instances from both datasets.
# MAGIC
# MAGIC 7. **Model Evaluation**:
# MAGIC    - Calculate and print the Silhouette Score for both training and testing data to evaluate the clustering quality.
# MAGIC
# MAGIC ### Evaluation Results:
# MAGIC
# MAGIC - The Silhouette Scores provide a measure of how well the data points have been clustered, with higher scores indicating better clustering quality.
# MAGIC

# COMMAND ----------

train_df, test_df = df.randomSplit([0.8, 0.2], seed=1)


assembler = VectorAssembler(inputCols=feature_columns, outputCol="feature")

kmeans = KMeans(k=5, seed=1, featuresCol="feature", predictionCol="cluster")

pipeline = Pipeline(stages=[assembler, kmeans])


model = pipeline.fit(train_df)


clustered_train_df = model.transform(train_df)
clustered_test_df = model.transform(test_df)


clustered_train_df.select("price", "cluster").show()
clustered_test_df.select("price", "cluster").show()

evaluator = ClusteringEvaluator(featuresCol='feature', predictionCol='cluster', metricName='silhouette')
silhouette_train = evaluator.evaluate(clustered_train_df)
print(f"Silhouette Score for Training Data: {silhouette_train}")


silhouette_test = evaluator.evaluate(clustered_test_df)
print(f"Silhouette Score for Testing Data: {silhouette_test}")



# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression Analysis with Cluster Feature in PySpark
# MAGIC
# MAGIC ### Process Overview:
# MAGIC
# MAGIC 1. **Data Splitting for Regression**:
# MAGIC    - Split the clustered training data into 80% training and 20% testing subsets using a consistent seed for reproducibility.
# MAGIC
# MAGIC 2. **Feature Assembly**:
# MAGIC    - Extend the feature set to include the cluster assignments as a new feature.
# MAGIC    - Use `VectorAssembler` to combine both original and cluster features into a single vector column for regression analysis.
# MAGIC
# MAGIC 3. **Regression Model Setup**:
# MAGIC    - Configure and fit a `LinearRegression` model using the assembled features, targeting `price` as the response variable.
# MAGIC
# MAGIC 4. **Model Prediction**:
# MAGIC    - Apply the trained regression model to the testing dataset to predict prices.
# MAGIC
# MAGIC 5. **Model Evaluation**:
# MAGIC    - Evaluate model performance using several metrics: Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R²).
# MAGIC
# MAGIC ### Evaluation Results:
# MAGIC
# MAGIC - **MAE**: Provides the average absolute difference between observed and predicted values, indicating overall prediction accuracy.
# MAGIC - **MSE and RMSE**: Measure the average squared difference and the square root of MSE, respectively, highlighting the variance in prediction errors.
# MAGIC - **R²**: Reflects the proportion of variance in the dependent variable predictable from the independent variables, indicating model fit quality.
# MAGIC
# MAGIC

# COMMAND ----------




train_df_reg, test_df_reg = clustered_train_df.randomSplit([0.8, 0.2], seed=1)
feature_columns_with_cluster = feature_columns + ["cluster"]

assembler_with_cluster = VectorAssembler(inputCols=feature_columns_with_cluster, outputCol="features_with_cluster")


train_assembled_df = assembler_with_cluster.transform(train_df_reg)
test_assembled_df = assembler_with_cluster.transform(test_df_reg)

lr = LinearRegression(featuresCol="features_with_cluster", labelCol="price")


lr_model = lr.fit(train_assembled_df)


predictions = lr_model.transform(test_assembled_df)

evaluator = RegressionEvaluator(labelCol="price", predictionCol="prediction")

mae = evaluator.evaluate(predictions, {evaluator.metricName: "mae"})
mse = evaluator.evaluate(predictions, {evaluator.metricName: "mse"})
rmse = evaluator.evaluate(predictions, {evaluator.metricName: "rmse"})
r2 = evaluator.evaluate(predictions, {evaluator.metricName: "r2"})

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# COMMAND ----------

lr_summary = lr_model.summary

print(f"Coefficients: {lr_model.coefficients}")
print(f"Intercept: {lr_model.intercept}")
print(f"R2: {lr_summary.r2}")
print(f"Adjusted R2: {lr_summary.r2adj}")
print(f"RMSE: {lr_summary.rootMeanSquaredError}")
print(f"MSE: {lr_summary.meanSquaredError}")
print(f"MAE: {lr_summary.meanAbsoluteError}")
