from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MovieRecommendationSystem") \
    .getOrCreate()

# Load the dataset
data = spark.read.csv('ratings.csv', header=True, inferSchema=True)

# Split the data into training and test sets
(train, test) = data.randomSplit([0.8, 0.2])

# Build the recommendation model using ALS
als = ALS(
    maxIter=10,
    regParam=0.1,
    userCol="userId",
    itemCol="movieId",
    ratingCol="rating",
    coldStartStrategy="drop"
)

# Train the model
model = als.fit(train)

# Evaluate the model
evaluator = RegressionEvaluator(
    metricName="rmse",
    labelCol="rating",
    predictionCol="prediction"
)
predictions = model.transform(test)
rmse = evaluator.evaluate(predictions)
print(f"Root-mean-square error = {rmse}")

# Save the model
model.save("movie_recommendation_model")

# Stop the Spark session
spark.stop()
