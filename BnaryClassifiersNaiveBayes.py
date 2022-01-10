from pyspark.sql import SparkSession
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder.appName("Naive Bayes Binary Classifier").getOrCreate()

data = spark.read.parquet(
    "C:/Users/talk2/Documents/unsw_clean_data.del", 
    inferSchema=True, header=True).select("pcaFeatures", "label")

train, test = data.randomSplit([0.7, 0.3], 25)

classify = NaiveBayes(featuresCol='pcaFeatures', labelCol='label', modelType="gaussian")
model = classify.fit(train)

prediction = model.transform(test)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="prediction", labelCol="label")

accuracy = evaluator.evaluate(prediction)
print('Naive Bayes Accuracy: {}'.format(accuracy))
# Prints out: Naive Bayes Accuracy: 0.9750322092256697

parameterGrid = ParamGridBuilder().build()

# Train the cross validator estimator to the training data
crossValidator = CrossValidator(estimator=classify, estimatorParamMaps=parameterGrid, evaluator=BinaryClassificationEvaluator(), numFolds=3)

nb_cvm = crossValidator.fit(train)
nb_cv_predicion = nb_cvm.transform(test)

# Evaluate the cross validation model using the BinaryClassificationEvaluator
result = evaluator.evaluate(nb_cv_predicion)
print()
print("Cross Validation Accuracy Score for NB Classifier: {}".format(result))
# Prints out: Cross Validation Accuracy Score for NB Classifier: 0.9750322092256697

