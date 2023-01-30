# Here is an example of code to create a simple neural network using the Keras library in Python
# to classify the penguin species

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense

# Load the data
penguin_df = pd.read_csv(r"data/penguin_raw_data.csv")

# Clean the data (drop NAs, clean up strings and change data types)
before_nadropped = penguin_df.shape[0]
penguin_df = penguin_df.dropna()
difference = before_nadropped - penguin_df.shape[0]
print(f"{difference} rows have been dropped")

penguin_df["Species"] = penguin_df["Species"].str.replace(" ", "")
penguin_df["Species"] = penguin_df["Species"].str.split("(").str[0]
penguin_df["Species"] = penguin_df["Species"].astype("category")
data_types = penguin_df.dtypes
print(data_types)

# Further preprocess the data to make suitable for neural networks (all variables must be numeric)
from sklearn.preprocessing import LabelEncoder

pen_df_numeric = penguin_df
for (column) in pen_df_numeric:  # iterate over columns, converting to numeric is needed
    if pen_df_numeric[column].dtype == "object" or "category":
        # Fit the LabelEncoder to the column and transform it
        encoder = LabelEncoder()
        pen_df_numeric[column] = encoder.fit_transform(pen_df_numeric[column])
print(pen_df_numeric.dtypes)


# Split the data into predictor and outcome variables
predictors = penguin_df.iloc[:, 3:]
outcome = penguin_df["Species"]

# Further seperate the data into training and test sets
predictors_train, predictors_test, outcome_train, outcome_test = train_test_split(
    predictors, outcome, test_size=0.2, random_state=3
)

# Normalise the input data
scaler = StandardScaler()
predictors_train = scaler.fit_transform(predictors_train)
predictors_test = scaler.transform(predictors_test)

# Create a model and compile the neural network
penguin_nn = Sequential()
penguin_nn.add(Dense(10, activation="relu", input_dim=predictors.shape[1]))
penguin_nn.add(Dense(10, activation="relu"))
penguin_nn.add(Dense(10, activation="relu"))
penguin_nn.add(Dense(3, activation="softmax"))
penguin_nn.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Fit the model on the training data
history = model.fit(
    predictors_train,
    outcome_train_train,
    epochs=100,
    batch_size=32,
    validation_data=(predictors_test_test, outcome_test_test),
)

