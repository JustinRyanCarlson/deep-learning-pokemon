import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense

pd.set_option('display.max_columns', None)

df = pd.read_csv('./Pokemon.csv')
df.drop(columns=['#', 'Name', 'Type 2'], inplace=True)

print('df shape: ' + str(df.shape))
print(df.describe())
print(df.columns)

encoder = LabelEncoder()
encoder.fit(df.Legendary)
encoded_targets = encoder.transform(df.Legendary)

type_encoder = LabelEncoder()
type_encoder.fit(df['Type 1'])
encoded_type1 = type_encoder.transform(df['Type 1'])
# encoded_type2 = type_encoder.transform(df['Type 2'])
df['Type 1'] = encoded_type1
# df['Type 2'] = encoded_type2

print(encoded_targets)
df.drop(columns='Legendary', inplace=True)
print(df)
n_cols = df.shape[1]

model = Sequential()
model.add(Dense(10, activation='relu', input_shape=(n_cols,)))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print(df.values)
model.fit(df.values, encoded_targets, epochs=10)