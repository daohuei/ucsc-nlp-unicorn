import pandas as pd
from rouge import Rouge

rouge = Rouge()


def calculate_rouges(preds, golds):
    result = {}
    scores = rouge.get_scores(preds, golds)
    score_df = pd.DataFrame(scores)
    for k in ["1", "2", "l"]:
        for m in ["p", "r", "f"]:
            key = f"rouge-{k}-{m}"
            value = (
                score_df[f"rouge-{k}"]
                .apply(lambda score_dict: score_dict[m])
                .mean()
            )
            result[key] = value

    return result


if __name__ == "__main__":
    preds = ["Hello Word", "Giu and Giao"]
    golds = ["Hello World", "Giuhun and Giao"]
    result = calculate_rouges(preds, golds)
    print(result)
    # rouge_score = calculate_rouge(test_data, SRC, TRG, model, device)

    # print("Rouge scores", np.array(rouge_score).mean(axis=0))
