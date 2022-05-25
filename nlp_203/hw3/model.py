from transformers import AdapterTrainer, Trainer


class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class QuestionAnsweringAdapterTrainer(
    QuestionAnsweringTrainer, AdapterTrainer
):
    pass

