import argparse

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.sql import SparkSession


def transform_data(output_uri: str) -> None:
    with SparkSession.builder.appName("Twitter Impressions").getOrCreate() as spark:

            # Load datasets
            training_data = spark.read.csv("s3://twitterimpressions/data/training.csv", header=True, inferSchema=True)
            validation_data = spark.read.csv("s3://twitterimpressions/data/validation.csv", header=True, inferSchema=True)
            test_data = spark.read.csv("s3://twitterimpressions/data/test.csv", header=True, inferSchema=True)

            # Preprocess the data
            # Convert labels to numeric format
            label_indexer = StringIndexer(inputCol="label", outputCol="label_index")

            # Tokenize text data
            tokenizer = Tokenizer(inputCol="text", outputCol="words")

            # Remove stop words
            stopwords_remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")

            # Convert text data into feature vectors
            count_vectorizer = CountVectorizer(inputCol="filtered_words", outputCol="raw_features")

            # Compute TF-IDF values
            idf = IDF(inputCol="raw_features", outputCol="features")

            # Build the classification model
            lr = LogisticRegression(featuresCol="features", labelCol="label_index", maxIter=10)

            # Create the pipeline
            pipeline = Pipeline(stages=[label_indexer, tokenizer, stopwords_remover, count_vectorizer, idf, lr])



            # Train the model
            model = pipeline.fit(training_data)

            # Validate the model
            validation_predictions = model.transform(validation_data)

            # Evaluate the model
            evaluator = MulticlassClassificationEvaluator(labelCol="label_index", predictionCol="prediction",
                                                          metricName="accuracy")


            accuracy = evaluator.evaluate(validation_predictions)
            print(f"Validation Accuracy: {accuracy}")

            # Test the model
            test_predictions = model.transform(test_data)
            test_accuracy = evaluator.evaluate(test_predictions)
            print(f"Test Accuracy: {test_accuracy}")

            # Save the model
            model.write().overwrite().save(output_uri)

            # Stop the Spark session
            spark.stop()

if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        parser.add_argument('--output_uri')
        args = parser.parse_args()

        transform_data(args.output_uri)