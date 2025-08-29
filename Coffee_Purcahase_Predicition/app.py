import pandas as pd 
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt 
import streamlit as st

data = {
    'Weather': ['Sunny', 'Rainy', 'Overcast', 'Sunny', 'Rainy', 'Sunny', 'Overcast', 'Rainy', 'Sunny', 'Rainy'],
    'TimeOfDay': ['Morning', 'Morning', 'Afternoon', 'Afternoon', 'Evening', 'Morning', 'Morning', 'Afternoon', 'Evening', 'Morning'],
    'SleepQuality': ['Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Poor', 'Good', 'Good', 'Poor'],
    'Mood': ['Tired', 'Fresh', 'Tired', 'Energetic', 'Tired', 'Fresh', 'Tired', 'Tired', 'Energetic', 'Tired'],
    'BuyCoffee': ['Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes']
}
df = pd.DataFrame(data)


df_encoded = df.copy()
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df_encoded[column] = le.fit_transform(df[column])
    label_encoders[column] = le

X = df_encoded.drop('BuyCoffee', axis=1)
y = df_encoded['BuyCoffee']

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X, y)

st.title("Coffee Purchase Predictor")
st.write("Predict whether a customer will buy coffee based on weather and personal conditions.")

def user_input():
    Weather = st.selectbox("Weather", df['Weather'].unique())
    TimeOfDay = st.selectbox("Time of Day", df['TimeOfDay'].unique())
    SleepQuality = st.selectbox("Sleep Quality", df['SleepQuality'].unique())
    Mood = st.selectbox("Mood", df['Mood'].unique())
    return pd.DataFrame([[Weather, TimeOfDay, SleepQuality, Mood]],
    columns=['Weather', 'TimeOfDay', 'SleepQuality', 'Mood'])

input_df = user_input()

input_encoded = input_df.copy()
for col in input_encoded.columns:
    input_encoded[col] = label_encoders[col].transform(input_encoded[col])

prediction = model.predict(input_encoded)[0]
predicted_label = label_encoders['BuyCoffee'].inverse_transform([prediction])[0]

st.subheader("Prediction:")
st.success(f"The model predicts: {predicted_label}")

st.subheader("Decision Tree Visualization")
fig, ax = plt.subplots(figsize=(10, 6))
plot_tree(model, feature_names=X.columns, class_names=label_encoders['BuyCoffee'].classes_, filled=True)
st.pyplot(fig)