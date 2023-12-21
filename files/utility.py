from pyspark.sql import SparkSession
from pyspark.sql.functions import col, month, year
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class AccidentPlotter:
    def __init__(self, df, predictions):
        self.df = df
        self.predictions = predictions

    def plot_accidents(self):
        # Group by year and month, and count occurrences
        result_df_pa = self.df.groupBy("year", "month").count()

        # Filter for true positive predictions
        true_predictions = self.predictions.filter(self.predictions['Severity'] == self.predictions['prediction'])

        # Group by year and month, and count true positive occurrences
        true_pred_count = true_predictions.groupBy("year", "month").count()

        # Collect the data to a pandas DataFrame for visualization
        pandas_true_pred = true_pred_count.toPandas().sort_values(by=["year", "month"])
        pandas_df_pa = result_df_pa.toPandas().sort_values(by=["year", "month"])

        # Create a continuous time scale
        pandas_true_pred['continuous_time'] = (pandas_true_pred['year'] - 2017) * 12 + pandas_true_pred['month']
        pandas_df_pa['continuous_time'] = (pandas_df_pa['year'] - 2017) * 12 + pandas_df_pa['month']

        # Styling with seaborn
        sns.set(style="whitegrid")

        # Plot
        fig, ax = plt.subplots(figsize=(20, 6))

        # Plotting lines for actual and true positive predicted counts
        range_year = [2017, 2022]
        year_df = pandas_df_pa[(pandas_df_pa['year'] >= range_year[0]) & (pandas_df_pa['year'] <= range_year[1])]
        ax.plot(year_df['continuous_time'], year_df['count'], marker='x', color="black", label=f"Actual from {range_year[0]} to {range_year[1]}")

        range_year2 = [2019, 2022]
        year_df = pandas_true_pred[(pandas_true_pred['year'] >= range_year2[0]) & (pandas_true_pred['year'] <= range_year2[1])]
        ax.plot(year_df['continuous_time'], year_df['count'], marker='x', color='red', label=f"True Predictions from {range_year2[0]} to {range_year2[1]}")

        # plot settings
        plt.xlabel('Time (2017-2022)', fontsize=14)
        plt.ylabel('Count of Accidents', fontsize=14)
        plt.title('Continuous Accident and True Predictions in Pennsylvania (2017-2022)', fontsize=16)

        xticks_labels = [f'{y}-{m:02d}' for y in range(2017, 2023) for m in range(1, 13)]
        plt.xticks(np.arange(1, len(xticks_labels) + 1, step=3), xticks_labels[::3], rotation=45, fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(loc='upper left')
        ax.yaxis.grid(True)

        plt.savefig('continuous_pa_accident_true_pred_count.png', bbox_inches='tight')
        plt.show()

    def plot_confusion_matrix(self):
        # Convert the Spark DataFrame to a Pandas DataFrame
        predictions_pd = self.predictions.select("prediction", "Severity").toPandas()

        # Generate the confusion matrix
        cm = confusion_matrix(predictions_pd['Severity'], predictions_pd['prediction'])

        # Plotting using seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual Label')
        plt.xlabel('Predicted Label')
        plt.show()