import gradio as gr
import numpy as np
import joblib
import skfuzzy as fuzz

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

x_prob = np.arange(0, 1.01, 0.01)

low_membership = fuzz.trimf(x_prob, [0, 0, 0.5])
medium_membership = fuzz.trimf(x_prob, [0.2, 0.5, 0.8])
high_membership = fuzz.trimf(x_prob, [0.5, 1, 1])

def fuzzy_risk(probabilities):

    labels = ["Low Risk", "Medium Risk", "High Risk"]
    output = ""

    for i, prob in enumerate(probabilities):

        low_val = fuzz.interp_membership(x_prob, low_membership, prob)
        med_val = fuzz.interp_membership(x_prob, medium_membership, prob)
        high_val = fuzz.interp_membership(x_prob, high_membership, prob)

        output += f"\n{labels[i]}\n"
        output += f"Probability: {prob:.3f}\n"
        output += f"Low Membership: {low_val:.3f}\n"
        output += f"Medium Membership: {med_val:.3f}\n"
        output += f"High Membership: {high_val:.3f}\n"

    return output

def predict(*inputs):

    arr = np.array(inputs).reshape(1,-1)
    arr_scaled = scaler.transform(arr)

    probs = model.predict_proba(arr_scaled)[0]

    fuzzy_output = fuzzy_risk(probs)

    return fuzzy_output


iface = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label=col) for col in columns],
    outputs="text",
    title="Cancer Risk Predictor with Fuzzy Explanation"
)

iface.launch()
