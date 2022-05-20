from gpu import device

BATCH_SIZE = 16
DEVICE = device

PROMPTS = [
    ("", ", my emotion is [MASK]."),
    ("I feel frustrated, my emotion is sadness.", ", my emotion is [MASK].",),
    (
        "I feel frustrated, my emotion is sadness. I feel happy, my emotion is joy.  I feel surprised, my emotion is surprise. I feel scared, my emotion is fear. ",
        ", my emotion is [MASK].",
    ),
    (
        "I feel frustrated, my emotion is sadness. I feel happy, my emotion is joy.  I feel surprised, my emotion is surprise. I feel scared, my emotion is fear. I feel angry, my emotion is anger. I feel loved, my emtion is love.",
        ", my emotion is [MASK].",
    ),
    ("", ". I feel [MASK]."),
]
PROMPT_CHOICE = 4

NAME = f"prompt_{PROMPT_CHOICE}"
PRE_PROMPT = PROMPTS[PROMPT_CHOICE][0]
SUF_PROMT = PROMPTS[PROMPT_CHOICE][1]
