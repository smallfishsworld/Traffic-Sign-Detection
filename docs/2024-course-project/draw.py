import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv("E:\笔记本-2024秋\机器学习\Project\code\\results.csv")  


def plot_metrics(data, x_column, y_columns, title, xlabel, ylabel, legend=True):
    plt.figure(figsize=(10, 6))
    for y_column in y_columns:
        plt.plot(data[x_column], data[y_column], label=y_column)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if legend:
        plt.legend()
    plt.grid(True)
    plt.show()


plot_metrics(
    data, 
    x_column="epoch", 
    y_columns=["train/box_loss", "val/box_loss", "train/cls_loss", "val/cls_loss", "train/dfl_loss", "val/dfl_loss"],
    title="Training and Validation Losses over Epochs",
    xlabel="Epoch",
    ylabel="Loss"
)


plot_metrics(
    data,
    x_column="epoch",
    y_columns=["metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"],
    title="Precision, Recall, and mAP Metrics over Epochs",
    xlabel="Epoch",
    ylabel="Metric Value"
)


plot_metrics(
    data,
    x_column="epoch",
    y_columns=["lr/pg0", "lr/pg1", "lr/pg2"],
    title="Learning Rates over Epochs",
    xlabel="Epoch",
    ylabel="Learning Rate"
)