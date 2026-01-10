import numpy as np

def evaluate_threshold(model, ds, threshold=0.15):
    dices, ious = [], []

    for x, y in ds:
        p = model.predict(x, verbose=0)
        p = (p > threshold).astype(np.float32)
        y = y.numpy()

        inter = np.sum(y * p)
        dice = (2 * inter) / (np.sum(y) + np.sum(p) + 1e-6)
        iou = inter / (np.sum(y) + np.sum(p) - inter + 1e-6)

        dices.append(dice)
        ious.append(iou)

    return float(np.mean(dices)), float(np.mean(ious))
