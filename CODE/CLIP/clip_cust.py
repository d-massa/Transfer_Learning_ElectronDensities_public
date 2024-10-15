import torch
import torch.nn as nn

class CLIPRegressionModel(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.regression_head = nn.Linear(self.clip_model.config.projection_dim*2, 1)

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip_model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Extract the text and image embeddings and concatenate them
        text_embeddings = outputs.text_embeds
        image_embeddings = outputs.image_embeds
        embeddings = torch.cat((text_embeddings, image_embeddings), dim=-1)
        return self.regression_head(embeddings).squeeze(-1)

def r2_score(y_true, y_pred):
    y_true_mean = torch.mean(y_true)
    ss_total = torch.sum((y_true - y_true_mean) ** 2)
    ss_residual = torch.sum((y_true - y_pred) ** 2)
    r2 = 1 - ss_residual / ss_total
    return r2.item()
