from flask import Flask, request, jsonify
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel

app = Flask(__name__)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("MovieRecommendationSystem") \
    .getOrCreate()

# Load the pre-trained model
model = ALSModel.load("movie_recommendation_model")

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('userId'))
    user_df = spark.createDataFrame([(user_id,)], ["userId"])
    
    # Get recommendations for the user
    recommendations = model.recommendForUserSubset(user_df, 10).collect()
    if recommendations:
        movie_recommendations = recommendations[0].recommendations
        return jsonify([row.movieId for row in movie_recommendations])
    else:
        return jsonify([])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
