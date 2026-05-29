import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import gradio as gr
import matplotlib.pyplot as plt
import cv2

# MODEL LOADING

model = load_model("skin_model1.h5", compile=False)

# Force model build to prevent GradCAM crash
dummy = np.zeros((1,224,224,3))
model(dummy)

classes = ["Acne","Eczema","Herpes","Panu","Rosacea"]

# IMAGE PREPROCESSING

def preprocess(img):

    img = img.resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img, axis=0)

    return img


# VITAMIN D SYMPTOM WEIGHTING SYSTEM

def vitamin_d_score(symptoms):

    if symptoms is None:
        symptoms = []

    weights = {
        "Fatigue":1,
        "Bone pain":3,
        "Muscle weakness":3,
        "Hair thinning":1,
        "Frequent illness":2,
        "Dull skin":1
    }

    score = 0

    for s in symptoms:
        score += weights.get(s,0)

    if score >= 6:
        risk = "HIGH Vitamin D Deficiency Risk. Blood Test recommended."

    elif score >= 3:
        risk = "MODERATE Vitamin D Deficiency Risk. Blood Test recommended."

    else:
        risk = "LOW Vitamin D Deficiency Risk. Blood Test optional."

    return score, risk


# GRAD-CAM HEATMAP

def gradcam_heatmap(img_array):

    try:

        last_conv_layer = None

        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.outputs]
        )

        with tf.GradientTape() as tape:

            conv_outputs, predictions = grad_model(img_array)

            class_idx = tf.argmax(predictions[0])

            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)

        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]

        heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

        heatmap = np.maximum(heatmap,0)

        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)

        return heatmap

    except:
        return None


# HEATMAP OVERLAY

def overlay_heatmap(original_img, heatmap):

    if heatmap is None:
        return original_img

    heatmap = cv2.resize(heatmap,(224,224))

    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.resize(np.array(original_img),(224,224))

    superimposed = cv2.addWeighted(original,0.6,heatmap,0.4,0)

    return superimposed


# PROBABILITY GRAPH

def probability_graph(predictions):

    predictions = np.squeeze(predictions)

    fig, ax = plt.subplots()

    ax.bar(classes, predictions)

    ax.set_title("Disease Probability Distribution")

    ax.set_xlabel("Skin Diseases")

    ax.set_ylabel("Probability")

    fig.tight_layout()

    return fig


# MAIN DETECTION FUNCTION

def detect(img, symptoms):

    if img is None:
        return "Please upload an image.", None, None

    img_array = preprocess(img)

    predictions = model.predict(img_array, verbose=0)

    predicted_class = classes[np.argmax(predictions)]

    confidence = np.max(predictions)

    score, risk = vitamin_d_score(symptoms)

    heatmap = gradcam_heatmap(img_array)

    heatmap_img = overlay_heatmap(img, heatmap)

    graph = probability_graph(predictions)

    result = f"""
Predicted Disease: {predicted_class}

Confidence: {confidence:.2f}

Vitamin D Symptom Score: {score}

Risk Level: {risk}
"""

    return result, heatmap_img, graph


# DISEASE INFORMATION

disease_info = """
### Skin Diseases

Acne – inflammatory condition caused by clogged pores.

Eczema – chronic inflammatory skin disorder causing itching and redness.

Herpes – viral infection producing blisters on the skin.

Panu – fungal infection causing discolored patches.

Rosacea – facial redness and visible blood vessels.

---

### Vitamin D Deficiency Symptoms

• Fatigue  
• Bone Pain  
• Muscle Weakness  
• Hair Thinning  
• Frequent Illness  
• Dull Skin
"""


# GRADIO INTERFACE

with gr.Blocks(title="AI Healthcare Skin Analysis Platform") as demo:

    gr.Markdown("# AI Skin Disease & Vitamin D Screening")

    # TAB 1
    with gr.Tab("Detection"):

        img_input = gr.Image(type="pil", label="Upload Skin Image")

        symptoms = gr.CheckboxGroup(
            choices=[
                "Fatigue",
                "Bone pain",
                "Muscle weakness",
                "Hair thinning",
                "Frequent illness",
                "Dull skin"
            ],
            value=[],
            label="Select Symptoms (Optional)"
        )

        detect_btn = gr.Button("Analyze Skin")

        result_output = gr.Textbox(label="Diagnosis Result")

        heatmap_output = gr.Image(label="AI Attention Map (Grad-CAM)")

        graph_output = gr.Plot(label="Prediction Probability")

        detect_btn.click(
            detect,
            inputs=[img_input, symptoms],
            outputs=[result_output, heatmap_output, graph_output]
        )

    # TAB 2
    with gr.Tab("Symptoms Checker"):

        symptoms_check = gr.CheckboxGroup(
            choices=[
                "Fatigue",
                "Bone pain",
                "Muscle weakness",
                "Hair thinning",
                "Frequent illness",
                "Dull skin"
            ],
            value=[],
            label="Select Symptoms"
        )

        check_btn = gr.Button("Check Vitamin D Risk")

        risk_output = gr.Textbox(label="Risk Result")

        def check(symptoms):
            score, risk = vitamin_d_score(symptoms)
            return f"Score: {score}\n{risk}"

        check_btn.click(
            check,
            inputs=symptoms_check,
            outputs=risk_output
        )

    # TAB 3
    with gr.Tab("Disease Information"):

        gr.Markdown(disease_info)


demo.launch()
