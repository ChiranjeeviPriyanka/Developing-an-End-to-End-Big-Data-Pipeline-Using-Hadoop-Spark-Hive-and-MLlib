
# importing necessary libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, count
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import subprocess

# Starting Spark session
spark = SparkSession.builder.appName("ChurnPredictionWithVisuals").getOrCreate()

# Loading cleaned data from GCS bucket
data_path = "gs://bigdataassignment01/ecommerce_csv/cleaned_oct2019_output_csv/"
df = spark.read.option("header", "true").option("inferSchema", "true").csv(data_path)

# Creating new column churn (0 = purchase, 1 = churn)
df = df.withColumn("churn", when(col("event_type") == "purchase", 0).otherwise(1))

# Aggregate features per user_session
session_features = df.groupBy("user_session").agg(
    count(when(col("event_type") == "view", True)).alias("views"),
    count(when(col("event_type") == "cart", True)).alias("cart_adds"),
    count(when(col("event_type") == "purchase", True)).alias("purchases"),
    count("*").alias("total_events"),
    count(when(col("churn") == 1, True)).alias("churn_label")
)

# Creating new column named Label: 1 if there is no purchase, else 0
final_df = session_features.withColumn("label", when(col("purchases") > 0, 0).otherwise(1))

# Assemble features vector
assembler = VectorAssembler(
    inputCols=["views", "cart_adds", "total_events"],
    outputCol="features"
)
assembled_df = assembler.transform(final_df).select("features", "label")

# Split the train(0.8) and test(0.2) sets for ML 
train, test = assembled_df.randomSplit([0.8, 0.2], seed=42)

# Applying Logistic Regression Machine learning algorithm + Cross-validation
lr = LogisticRegression()
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.1]).build()
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid,
                    evaluator=BinaryClassificationEvaluator(), numFolds=3)
model = cv.fit(train)

# Predicting on test set
predictions = model.transform(test)

# Evaluate machine learning algorithm using various metrics
binary_eval = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")

auc = binary_eval.evaluate(predictions, {binary_eval.metricName: "areaUnderROC"})

multi_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction")

accuracy = multi_eval.evaluate(predictions, {multi_eval.metricName: "accuracy"})
precision = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedPrecision"})
recall = multi_eval.evaluate(predictions, {multi_eval.metricName: "weightedRecall"})
f1 = multi_eval.evaluate(predictions, {multi_eval.metricName: "f1"})

print(f"Metrics:\nAUC: {auc:.4f}\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-score: {f1:.4f}")

# Sample predictions to Pandas for visualization
pred_pd = predictions.select("label", "prediction", "probability").sample(False, 0.1, seed=42).toPandas()

# Extract probability of positive class (churn=1)
pred_pd['prob_churn'] = pred_pd['probability'].apply(lambda x: float(x[1]))

# Plot ROC Curve for visualizing results
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix

fpr, tpr, thresholds = roc_curve(pred_pd['label'], pred_pd['prob_churn'])
roc_auc = roc_auc_score(pred_pd['label'], pred_pd['prob_churn'])

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
roc_path_local = "/tmp/roc_curve.png"
plt.savefig(roc_path_local)
plt.close()

# Plot Confusion Matrix for visualizing results (using threshold 0.5)
y_true = pred_pd['label']
y_pred = (pred_pd['prob_churn'] >= 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["No Churn","Churn"], yticklabels=["No Churn","Churn"])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
cm_path_local = "/tmp/confusion_matrix.png"
plt.savefig(cm_path_local)
plt.close()

# Uploading both plots to GCS bucket
output_gcs_path = "gs://bigdataassignment01/results/churn_predictions/plots/"

subprocess.run(["gsutil", "cp", roc_path_local, output_gcs_path + "roc_curve.png"], check=True)
subprocess.run(["gsutil", "cp", cm_path_local, output_gcs_path + "confusion_matrix.png"], check=True)

print("ROC curve and Confusion Matrix saved to GCS!")

# Stop Spark session
spark.stop()
