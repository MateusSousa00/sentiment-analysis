import joblib
import pandas as pd

baseline_results = joblib.load("src/models/baseline_model/evaluation_results.pkl")
fine_tuned_results = joblib.load("src/models/transformer_finetuned/evaluation_results.pkl")

df = pd.DataFrame({
    "Metric": ["Accuracy","Precision","Recall","F1-score"],
    "Baseline Model": [
        baseline_results["accuracy"],
        baseline_results["precision"],
        baseline_results["recall"],
        baseline_results["f1_score"],
    ],
    "Fine-Tuned Model": [
        fine_tuned_results["accuracy"],
        fine_tuned_results["precision"],
        fine_tuned_results["recall"],
        fine_tuned_results["f1_score"],
    ]
})

print("\n Model Performance Comparison: \n")
print(df.to_string(index=False))

df.to_csv("src/models/comparison_results.csv", index=False)

print("\n Comparison results saved to `src/models/comparison_results.csv`!")