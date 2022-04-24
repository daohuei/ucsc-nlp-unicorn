import pandas as pd


def print_stage(stage_str):
    count = 100
    occupied_count = len(stage_str)
    separator_num = int((count - occupied_count) / 2)
    separator_str = "=" * separator_num
    print_str = f"{separator_str}{stage_str}{separator_str}"
    print(print_str)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def write_predictions(preds, split, name):
    preds = [" ".join(pred) for pred in preds]
    with open(f"./{name}.{split}.pred", "w") as f:
        f.write("\n".join(preds))


def write_scores(scores, split, name):
    df = pd.DataFrame(scores)
    df.to_csv(f"./{name}_{split}_score.csv", index=False)
