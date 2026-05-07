import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score


def accuracy_at_k(output: torch.Tensor, target: torch.Tensor, topk=(1, 3)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    results = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0).item()
        results.append(correct_k)
    return results


def evaluate_model(model, dataloader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    top1_correct = 0.0
    top3_correct = 0.0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            c1, c3 = accuracy_at_k(outputs, labels, topk=(1, 3))
            top1_correct += c1
            top3_correct += c3
            total += labels.size(0)

            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    top1_acc = top1_correct / total
    top3_acc = top3_correct / total
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")
    cm = confusion_matrix(all_labels, all_preds)

    per_class_acc = {}
    for idx, name in enumerate(class_names):
        per_class_acc[name] = cm[idx, idx] / np.sum(cm[idx]) if np.sum(cm[idx]) > 0 else 0.0

    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
    )

    return {
        "top1_acc": top1_acc,
        "top3_acc": top3_acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "confusion_matrix": cm,
        "per_class_acc": per_class_acc,
        "classification_report": report,
    }


def print_results(results, class_names) -> None:
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Top-1 Accuracy: {results['top1_acc']:.4f} ({results['top1_acc'] * 100:.2f}%)")
    print(f"Top-3 Accuracy: {results['top3_acc']:.4f} ({results['top3_acc'] * 100:.2f}%)")
    print(f"F1-score (Macro): {results['f1_macro']:.4f}")
    print(f"F1-score (Weighted): {results['f1_weighted']:.4f}")

    print("\nPer-Class Accuracy:")
    for name in class_names:
        acc = results["per_class_acc"][name]
        print(f"  {name:12s}: {acc:.4f} ({acc * 100:.2f}%)")
    print("=" * 60)
