from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
from net.utils.label_smooth import LabelSmoothLoss
class BertForSequenceClassificationLabelSmooth(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.args = args
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.label_smooth_loss = LabelSmoothLoss(args.label_smooth)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            #print("$"*10+'label存在'+"$"*10)
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                sigmoid_layer = nn.Sigmoid()
                logits = sigmoid_layer(logits)
                loss_fct = BCELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
                #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                #loss = self.label_smooth_loss(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (logits,) + outputs[2:] 
            outputs = (loss,) + outputs
            
        return outputs  # (loss), logits, (hidden_states), (attentions)