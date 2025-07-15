import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("model_ready_dataset.csv")
df["time_of_day"] = df["time_of_day"].astype("category").cat.codes
X = df[["steps_last_30min", "time_of_day"]]
y = df["level"]
rf_model = RandomForestRegressor(n_estimators=500, random_state=42)
rf_model.fit(X, y)


st.set_page_config(page_title="Glucose Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Glucose Predictor")
st.write("""
Predict your glucose level based on your walking activity.  
More walking = lower glucose prediction!
""")

if "history" not in st.session_state:
    st.session_state.history = []

baseline_glucose = st.number_input("Your typical fasting glucose (mg/dL)", min_value=70, max_value=300, value=180)


steps = st.number_input("Steps in the last 30 minutes", min_value=0, max_value=20000, value=0, step=100)

# time of day
time_of_day = st.selectbox(
    "Time of Day",
    options=["Morning", "Afternoon", "Evening", "Night"]
)
time_of_day_map = {"Morning": 0, "Afternoon": 1, "Evening": 2, "Night": 3}
time_of_day_code = time_of_day_map[time_of_day]

recent_meal = st.checkbox("Did you eat in the past 2 hours?")

# Predict
if st.button("Predict"):
    
    input_data = pd.DataFrame({
        "steps_last_30min": [steps],
        "time_of_day": [time_of_day_code]
    })
    prediction = rf_model.predict(input_data)[0]

    
    adjustment = 0.002 * steps
    prediction_adjusted = prediction - adjustment

    if recent_meal:
        prediction_adjusted += 20

    prediction_adjusted = max(prediction_adjusted, 70)

    st.success(f"Predicted Glucose Level: {prediction_adjusted:.1f} mg/dL")

    st.session_state.history.append({
        "Steps": steps,
        "Time of Day": time_of_day,
        "Meal": "Yes" if recent_meal else "No",
        "Prediction": f"{prediction_adjusted:.1f} mg/dL"
    })

    if prediction_adjusted > 140:
        st.info("🔔 Consider additional walking or adjusting your next meal.")
    elif prediction_adjusted < 90:
        st.warning("⚠️ Your predicted glucose is on the lower side.")
    else:
        st.success("✅ Your predicted glucose is in a good range!")

if st.session_state.history:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history))

#prediction trend chart
st.subheader("Estimated Glucose vs. Steps Trend")
step_range = np.arange(0, 20000, 500)
predictions_trend = []
for s in step_range:
    input_data = pd.DataFrame({
        "steps_last_30min": [s],
        "time_of_day": [time_of_day_code]
    })
    pred = rf_model.predict(input_data)[0]
    adj_pred = max(pred - 0.002 * s, 70)
    predictions_trend.append(adj_pred)

chart_df = pd.DataFrame({
    "Steps": step_range,
    "Predicted Glucose": predictions_trend
})
st.line_chart(chart_df.set_index("Steps"))

