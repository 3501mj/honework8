import gradio as gr
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PyAFAR_GUI.adult_afar import adult_afar

# 表情权重函数
def calculate_expression_scores(row):

    is_sad = row.get('Occ_au_15', 0) > row.get('Occ_au_12', 0) and row.get('Occ_au_17', 0) > row.get('Occ_au_12', 0)

    scores = {
        # Neutral
        "Neutral": 1.0 - min(
            1.0,
            row.get('Occ_au_12', 0) * 0.3 +
            row.get('Occ_au_15', 0) * 0.25 +
            row.get('Occ_au_17', 0) * 0.25 +
            row.get('Occ_au_6', 0) * 0.2 +
            row.get('Occ_au_4', 0) * 0.1
        ),

        # Happiness
        "Happiness": (row.get('Occ_au_12', 0) * 0.7 + row.get('Occ_au_6', 0) * 0.3),

        # Sadness
        "Sadness": (row.get('Occ_au_15', 0) * 0.8 + row.get('Occ_au_17', 0) * 0.2) if is_sad else 0,

        # Anger
        "Anger": (row.get('Occ_au_4', 0) * 0.6 + row.get('Occ_au_7', 0) * 0.4),

        # Surprise
        "Surprise": (row.get('Occ_au_1', 0) * 0.5 + row.get('Occ_au_2', 0) * 0.5),

        # Fear
        "Fear": (row.get('Occ_au_1', 0) * 0.35 + row.get('Occ_au_4', 0) * 0.4 + row.get('Occ_au_7', 0) * 0.25)
    }
    return scores

def process_image(image_path):
    try:
        # 调用 PyAFAR 模型
        adult_result = adult_afar(
            filename=image_path,
            AUs=["au_1", "au_2", "au_4", "au_6", "au_7", "au_10", "au_12", "au_14", "au_15", "au_17", "au_23", "au_24"],
            GPU=False,
            max_frames=1,  # 只处理一帧
            AU_Int=["au_6", "au_10", "au_12", "au_14", "au_17"],
            batch_size=10,
            PID=False
        )
    except Exception as e:
        return None, None, None


    adult_df = pd.DataFrame.from_dict(adult_result)


    occ_columns = [f"Occ_au_{i}" for i in [6, 10, 12, 14, 17]]
    try:
        au_occ_values = adult_df.iloc[0][occ_columns].values
        x_points = adult_df.iloc[0][[f"x_{i}" for i in range(468)]].values
        y_points = adult_df.iloc[0][[f"y_{i}" for i in range(468)]].values
    except KeyError as e:
        return None, None, None


    image = cv2.imread(image_path)
    if image is None:
        return None, None, None


    fig1, ax1 = plt.subplots(figsize=(6, 6))
    image_height, image_width, _ = image.shape
    x_points = (x_points * image_width).astype(int)
    y_points = (y_points * image_height).astype(int)
    ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax1.scatter(x_points, y_points, c='yellow', s=5)
    ax1.set_title("Facial Landmark Visualizer", fontsize=14)
    ax1.axis('off')
    fig1.canvas.draw()
    landmark_visual = fig1


    fig2, ax2 = plt.subplots(figsize=(6, 4))
    au_labels = [f"AU_{i}" for i in [6, 10, 12, 14, 17]]
    ax2.barh(au_labels, au_occ_values, color='lightgreen')
    ax2.set_xlim(0, max(au_occ_values) * 1.1 if max(au_occ_values) > 0 else 1)
    ax2.set_ylim(-0.5, len(au_labels) - 0.5)
    ax2.set_xlabel("Occurrence", fontsize=12)
    ax2.set_ylabel("Action Units", fontsize=12)
    ax2.set_title("Action Unit Occurrence Visualizer", fontsize=14)
    ax2.grid(axis='x', linestyle='--', alpha=0.7)
    for i, v in enumerate(au_occ_values):
        ax2.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=10)
    fig2.canvas.draw()
    au_visual = fig2


    expression_scores = calculate_expression_scores(adult_df.iloc[0])
    expressions = list(expression_scores.keys())
    scores = list(expression_scores.values())


    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.barh(expressions, scores, color='lightgreen')
    ax3.set_xlim(0, 1)
    ax3.set_xlabel("Probability", fontsize=12)
    ax3.set_title("Expression Probabilities", fontsize=14)
    ax3.grid(axis='x', linestyle='--', alpha=0.7)
    for i, v in enumerate(scores):
        ax3.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=10)
    fig3.canvas.draw()
    expression_visual = fig3

    return landmark_visual, au_visual, expression_visual

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Image")
            landmark_output = gr.Plot(label="Facial Landmark Visualizer")

        # with gr.Column():
        #     pass

    with gr.Row():
        au_bar_output = gr.Plot(label="Action Unit Occurrence Visualizer")
        expression_output = gr.Plot(label="Expression Probabilities")

    with gr.Row():
        process_button = gr.Button("Process Image")


    process_button.click(
        process_image,
        inputs=image_input,
        outputs=[landmark_output, au_bar_output, expression_output]
    )


demo.launch()