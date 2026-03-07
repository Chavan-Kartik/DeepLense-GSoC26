import numpy as np
from sklearn.metrics import roc_curve, auc, classification_report
import matplotlib.pyplot as plt

def compute_metrics(y_true, y_probs, class_names):
    """
    Computes One-vs-Rest ROC AUC for multi-class classification.
    y_true: [N] (integer labels)
    y_probs: [N, 3] (softmax probabilities)
    """
    plt.figure(figsize=(10, 8))
    
    auc_scores = {}
    
    for i, class_name in enumerate(class_names):
        # Binary target for the current class
        y_binary = (np.array(y_true) == i).astype(int)
        # Probabilities for the current class
        y_score = np.array(y_probs)[:, i]
        
        fpr, tpr, _ = roc_curve(y_binary, y_score)
        roc_auc = auc(fpr, tpr)
        auc_scores[class_name] = roc_auc
        
        plt.plot(fpr, tpr, label=f'ROC {class_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Multi-Class One-vs-Rest ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig('notebooks/roc_curve.png')
    plt.show()
    
    return auc_scores