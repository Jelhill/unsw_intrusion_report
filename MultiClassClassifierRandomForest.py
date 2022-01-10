from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("PySpark Machine Learning: Binary Classification").getOrCreate()

# UNSW-NB15 DATASET: CLEANED VERSION
unsw_data = spark.read.csv("hdfs://localhost:9000/tmp/exported/clean_data/network_data_final",
                           inferSchema=True).toDF("srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes", "dbytes",
                                                  "sttl", "dttl", "sloss", "dloss", "service", "sload", "dload", "spkts", "dpkts",
                                                  "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
                                                  "sjit", "djit", "stime", "ltime", "sintpkt", "dintpkt", "tcprtt", "synack", "ackdat",
                                                  "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
                                                  "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                                                  "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label")

unsw_net_data = unsw_data.select("state", "service", "sttl", "swin", "dwin", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm",
                                 "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "label", "attack_cat")

from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml.feature import PCA

categorical_cols = ["state", "service", "attack_cat"]
indexed_cols = ["state_index", "service_index", "attack_cat_index"]

indexed_unsw_net_data = StringIndexer(inputCols=categorical_cols, 
                                      outputCols=indexed_cols).fit(unsw_net_data).transform(unsw_net_data)

vectorised_unsw_net_data = VectorAssembler(inputCols=["state_index", "service_index", "sttl", "swin", "dwin", "ct_srv_src",
                                                      "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
                                                      "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat_index"],
                                           outputCol="features").transform(indexed_unsw_net_data)

unsw_net_data_final = vectorised_unsw_net_data.select("features", "label", "attack_cat_index")

unsw_net_data_scaled = StandardScaler(inputCol="features", 
                                      outputCol="scaledFeatures",
                                      withStd=True, withMean=True).fit(unsw_net_data_final).transform(unsw_net_data_final)

# PCA DIMENSIONALITY REDUCTION
col_names = unsw_net_data_scaled.columns
features_rdd = unsw_net_data_scaled.rdd.map(lambda x:x[0:]).toDF(col_names)


pca = PCA(k=5, inputCol="scaledFeatures", outputCol="pcaFeatures")
pca_reduced_unsw_data = pca.fit(features_rdd).transform(features_rdd).select('pcaFeatures', 'label', 'attack_cat_index')

unsw_mc_data = pca_reduced_unsw_data.select('pcaFeatures', 'attack_cat_index').toDF('features','attack_cat_index')

unsw_mc_data.show(n=10, truncate=False)

# SETTING THE STAGE FOR MACHINE LEARNING
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
spark = SparkSession.builder.appName("Binary Classifier Logistic Regression").getOrCreate()

data = spark.read.parquet(
    "C:/Users/talk2/Documents/unsw_clean_data.del", 
    inferSchema=True, header=True).select("pcaFeatures", "label")

train, test = unsw_mc_data.randomSplit([0.7, 0.3], seed=25)
train.show(n=10, truncate=False)
test.show(n=10, truncate=False)

random_forest = RandomForestClassifier(labelCol="attack_cat_index", featuresCol="features")
model = random_forest.fit(train)

prediction = model.transform(test)

accuracy = MulticlassClassificationEvaluator(labelCol="attack_cat_index", predictionCol="prediction", metricName="accuracy")

random_forest_accuracy = accuracy.evaluate(prediction)
print("Accuracy Score: {}".format(random_forest_accuracy))

parameter_grid = ParamGridBuilder().addGrid(random_forest.numTrees, [5, 20, 50]).addGrid(random_forest.maxDepth, [2, 5, 10]).build()
cross_validate = CrossValidator(estimator=rf, estimatorParamMaps=parameter_grid, evaluator=accuracy, numFolds=3)
cross_validate_model = cross_validate.fit(train)
cross_validate_prediction = cross_validate_model.transform(test)

cross_validate_accuracy = accuracy.evaluate(cross_validate_prediction)
print("Accuracy Score: {}".format(cross_validate_accuracy))
# Prints out: Cross Validation Accuracy Score: 0.9771437414623839

