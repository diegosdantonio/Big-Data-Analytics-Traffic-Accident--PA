from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, year, hour, to_timestamp, window, avg, count
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier, NaiveBayes, LogisticRegression
from pyspark.sql.types import DoubleType, IntegerType, StringType

from pyspark.sql.functions import pandas_udf
import geopandas as gpd
import shapely
import pandas as pd
from shapely.geometry import box
import warnings

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

from pyspark.sql.functions import lit

warnings.simplefilter(action='ignore', category=FutureWarning)


class AccidentSeverityPredictor:
    def __init__(self, feature_cols, bool_columns, window_duration = "1 day" , slide_duration = "1 day"):
        self.spark = SparkSession.builder \
            .appName("PA Accident Analysis Continuous (2018-2021)") \
            .config('spark.executor.memory', '2g') \
            .config('spark.driver.memory', '8g') \
            .getOrCreate()

        # self.classifier = # Default classifier is set to RandomForestClassifier, but can be changed
        
        self.feature_cols = feature_cols
        self.bool_columns = bool_columns
        self.window_duration = window_duration
        self.slide_duration = slide_duration

        

    def load_and_preprocess_data(self, file_path):
        df_raw = self.spark.read.format("csv").option("header", "true").load(file_path)
        df_pa = df_raw.filter(df_raw["State"] == "PA")#.sample(fraction=0.001)
        
        # Data type conversions and feature extraction
        # Convert columns to appropriate types
        df_pa = df_pa.withColumn("Start_Lat", col("Start_Lat").cast(DoubleType()))
        df_pa = df_pa.withColumn("Start_Lng", col("Start_Lng").cast(DoubleType()))
        df_pa = df_pa.withColumn("Street", col("Street").cast(StringType()))
        df_pa = df_pa.withColumn("City", col("City").cast(StringType()))
        df_pa = df_pa.withColumn("Zipcode", col("Zipcode").cast(IntegerType()))
        df_pa = df_pa.withColumn("Country", col("Country").cast(StringType()))
        df_pa = df_pa.withColumn("Temperature(F)", col("Temperature(F)").cast(DoubleType()))
        df_pa = df_pa.withColumn("Visibility(mi)", col("Visibility(mi)").cast(DoubleType()))
        df_pa = df_pa.withColumn("Distance(mi)", col("Distance(mi)").cast(DoubleType()))
        df_pa = df_pa.withColumn("Severity", col("Severity").cast(IntegerType()))
        df_pa = df_pa.withColumn('Start_Time', to_timestamp(col('Start_Time')))
        
        # Extract time features
        df_pa = df_pa.withColumn("month", month("Start_Time"))
        df_pa = df_pa.withColumn("year", year("Start_Time"))
        df_pa = df_pa.withColumn("hour", hour("Start_Time"))

        
        # df_balanced = self.perform_negative_sampling(df_pa, sampling_fraction=0.1)

        # df_pa = df_pa.withColumn("Grid", find_grid_udf("Start_Lat", "Start_Lng"))

        df_pa = df_pa.na.drop()

        
        return df_pa

    def perform_negative_sampling(self, df, sampling_fraction=0.1):
        
        # Negative samples are where 'Severity' is 0
        negative_samples = df.filter(col("Severity") == 0)
        positive_samples = df.filter(col("Severity") != 0)

        sampled_negative = negative_samples.sample(False, sampling_fraction)
        return positive_samples.unionAll(sampled_negative)


    def create_windowed_dataframe(self, df_pa):
    # Create a windowed DataFrame
        windowed_df = df_pa.groupBy(window("Start_Time", self.window_duration, self.slide_duration), "Severity").agg(
            avg("Temperature(F)").alias("avg_temp"),
            avg("Visibility(mi)").alias("avg_visibility"),
            count("Severity").alias("count_severity")
        ).orderBy("window")
        
        # Alias the DataFrames
        df_pa_alias = df_pa.alias("df_pa")
        windowed_df_alias = windowed_df.alias("windowed_df")
        
        
        # Perform the join using aliases
        df_pa = df_pa_alias.join(
            windowed_df_alias, 
            col("df_pa.Start_Time") >= col("windowed_df.window.start")
        ).select(
            "df_pa.*",  # Select all columns from df_pa
            "windowed_df.avg_temp", 
            "windowed_df.avg_visibility", 
            "windowed_df.count_severity"
        )
        return df_pa
    ############################################################

    def randomforest(self, df_pa, numTrees):
        self.classifier = RandomForestClassifier(labelCol="Severity", featuresCol="features", numTrees=numTrees)

        self.train_data = df_pa.filter((df_pa["year"] >= 2017) & (df_pa["year"] <= 2019))
        self.test_data = df_pa.filter(df_pa["year"] >= 2020)
        
        self.model = self.prepare_features().fit(self.train_data)
        self.predictions = self.model.transform(self.test_data)

        evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction", metricName="accuracy")
        # evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
        # evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction", metricName="recallByLabel")


        # accuracy = evaluator.evaluate(self.predictions)
        accuracy = evaluator.evaluate(self.predictions)
        print("Test Accuracy = %g" % accuracy)

        return [self.predictions, accuracy]

    def logisticregression(self, df_pa, maxIter = 100):
        self.classifier = LogisticRegression(labelCol="Severity", featuresCol="features", maxIter=maxIter)

        self.train_data = df_pa.filter((df_pa["year"] >= 2017) & (df_pa["year"] <= 2019))
        self.test_data = df_pa.filter(df_pa["year"] >= 2020)
        
        self.model = self.prepare_features().fit(self.train_data)
        self.predictions = self.model.transform(self.test_data)


        evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction", metricName="accuracy")
        # evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="prediction")
        # evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction", metricName="recallByLabel")


        # accuracy = evaluator.evaluate(self.predictions)
        accuracy = evaluator.evaluate(self.predictions)
        print("Test Accuracy = %g" % accuracy)

        return [self.predictions, accuracy]

    def naivebayes(self,  df_pa, smoothing = 1):
        self.classifier = NaiveBayes(labelCol="Severity", featuresCol="features", smoothing=smoothing, modelType="multinomial")

        self.train_data = df_pa.filter((df_pa["year"] >= 2017) & (df_pa["year"] <= 2019))
        self.test_data = df_pa.filter(df_pa["year"] >= 2020)

        # Find the majority class in the training dataset
        majority_class = self.train_data.groupBy("Severity").count().orderBy("count", ascending=False).first()["Severity"]
    
        # Create predictions for the test dataset using the majority class
        self.predictions = self.test_data.withColumn("prediction", lit(majority_class).cast(DoubleType()))
    
        # Evaluate the ZeroR model
        evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(self.predictions)
        print("ZeroR Model Test Accuracy = %g" % accuracy)

        return [self.predictions, accuracy]
    
    ############################################################



    def logisticregression_cross(self, df_pa, maxIter=100):
        # Prepare the Logistic Regression Classifier
        self.classifier = LogisticRegression(labelCol="Severity", featuresCol="features", maxIter=maxIter)

        self.train_data = df_pa.filter((df_pa["year"] >= 2017) & (df_pa["year"] <= 2019))
        self.test_data = df_pa.filter(df_pa["year"] >= 2020)
        
        # Prepare the Pipeline
        pipeline =  self.prepare_features().fit(self.train_data)

        # Create a ParamGridBuilder for hyperparameter tuning
        paramGrid = ParamGridBuilder() \
            .addGrid(self.classifier.regParam, [0.1, 0.01]) \
            .build()

        # Create a CrossValidator
        crossval = CrossValidator(estimator=pipeline,
                                  estimatorParamMaps=paramGrid,
                                  evaluator=MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction", metricName="accuracy"),
                                  numFolds=5)  # Use 3+ folds in practice

        # Run cross-validation, and choose the best set of parameters.
        cvModel = crossval.fit(df_pa)

        # Fetch best model
        self.model = cvModel.bestModel

        # Make predictions on test data. cvModel uses the best model found.
        self.test_data = df_pa.filter(df_pa["year"] >= 2020)
        self.predictions = self.model.transform(self.test_data)

        # Evaluate the model
        evaluator = MulticlassClassificationEvaluator(labelCol="Severity", predictionCol="prediction", metricName="accuracy")
        accuracy = evaluator.evaluate(self.predictions)
        print("Test Accuracy = %g" % accuracy)

        return [self.predictions, accuracy]    
    
    def prepare_features(self):
        # Handle boolean columns

        indexers = [StringIndexer(inputCol=c, outputCol=c+"_index").setHandleInvalid("skip") for c in self.bool_columns]
        encoders = [OneHotEncoder(inputCol=c+"_index", outputCol=c+"_vec") for c in self.bool_columns]

        # Assemble features
        assembler = VectorAssembler(inputCols=self.feature_cols, outputCol="features")
        return Pipeline(stages=indexers + encoders + [assembler, self.classifier])




    def sparkstop(self):
        self.spark.stop()



def create_grid():    
    pa_gdf = gpd.read_file("PaCounty2023_10.geojson")
    pa_gdf = pa_gdf.set_index("COUNTY_NAM")
    
    # Load the US primary roads data
    pa_map = gpd.read_file('data/tl_2022_us_primaryroads/tl_2022_us_primaryroads.shx')
    
    # Create the grid
    xmin, ymin, xmax, ymax = pa_gdf.total_bounds
    width = height = 0.1  # size of the square in degrees, adjust as needed
    
    rows = int((ymax - ymin) / height)
    cols = int((xmax - xmin) / width)
    squares = []
    
    count = []
    k = 0
    for i in range(cols):
        for j in range(rows):
            minx = xmin + i * width
            maxx = minx + width
            miny = ymin + j * height
            maxy = miny + height
            
            k = k + 1
            squares.append(box(minx, miny, maxx, maxy))
            count.append(k)
    
    gdf_grid = gpd.GeoDataFrame({'geometry': squares, 'ID':count })
    return gdf_grid


@pandas_udf(DoubleType())
def find_grid_udf(latitudes, longitudes):
    gdf_grid = create_grid()
    
    points = gpd.GeoSeries(gpd.points_from_xy(longitudes, latitudes))
    grids = []  # Initialize the list outside the loop

    for point in points:
        grid = 0  # Initialize grid as None for each point
        for idx, row in gdf_grid.iterrows():
            if row['geometry'].intersects(point):
                grid = row['ID']
                break
        grids.append(grid)  # Append the found grid (or None) to the grids list

    return pd.Series(grids)

