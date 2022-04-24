import torch
from transformers import (
    AutoModelForTokenClassification,
    DistilBertModel,
    AlbertModel,
)

from data import slot_vocab, intent_vocab


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class DistilBERTSlotModel(torch.nn.Module):
    def __init__(self):
        super(DistilBERTSlotModel, self).__init__()
        self.distil_bert = AutoModelForTokenClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(slot_vocab)
        )

    def forward(self, **batch):
        output = self.distil_bert(**batch)
        return output


class DistilBERTIntentModel(torch.nn.Module):
    def __init__(self):
        super(DistilBERTIntentModel, self).__init__()
        self.distil_bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased"
        )
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(768, len(intent_vocab))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # only retrieve the [CLS] value
        cls = self.distil_bert(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        output = self.dropout(cls)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output


class ALBERTSlotModel(torch.nn.Module):
    def __init__(self):
        super(ALBERTSlotModel, self).__init__()
        self.albert = AutoModelForTokenClassification.from_pretrained(
            "albert-base-v2", num_labels=len(slot_vocab)
        )

    def forward(self, **batch):
        output = self.albert(**batch)
        return output


class ALBERTIntentModel(torch.nn.Module):
    def __init__(self):
        super(ALBERTIntentModel, self).__init__()
        self.albert = AlbertModel.from_pretrained("albert-base-v2")
        self.dropout = torch.nn.Dropout(0.5)
        self.linear = torch.nn.Linear(768, len(intent_vocab))
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        # only retrieve the [CLS] value
        cls = self.albert(
            input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]
        output = self.dropout(cls)
        output = self.linear(output)
        output = self.sigmoid(output)
        return output