from ast import arg
from data_process import *
from args import *
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def train(
        model: nn.Module,
        train_dl: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: LambdaLR,
        n_epochs: int,
        device: torch.device
):
    """
    The main training loop which will optimize a given model on a given dataset
    :param model: The model being optimized
    :param train_dl: The training dataset
    :param optimizer: The optimizer used to update the model parameters
    :param n_epochs: Number of epochs to train for
    :param device: The device to train on
    """

    # Keep track of the loss and best accuracy
    losses = []
    best_acc = 0.0
    pcounter = 0

    # Iterate through epochs
    for ep in range(n_epochs):

        loss_epoch = []

        # Iterate through each batch in the dataloader
        for batch in tqdm(train_dl):
            # VERY IMPORTANT: Make sure the model is in training mode, which turns on
            # things like dropout and layer normalization
            model.train()

            # VERY IMPORTANT: zero out all of the gradients on each iteration -- PyTorch
            # keeps track of these dynamically in its computation graph so you need to explicitly
            # zero them out
            optimizer.zero_grad()

            # Place each tensor on the GPU
            batch = {b: batch[b].to(device) for b in batch}

            # Pass the inputs through the model, get the current loss and logits
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                start_positions=batch['start_tokens'],
                end_positions=batch['end_tokens']
            )
            loss = outputs['loss']
            losses.append(loss.item())
            loss_epoch.append(loss.item())

            # Calculate all of the gradients and weight updates for the model
            loss.backward()

            # Optional: clip gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Finally, update the weights of the model and advance the LR schedule
            optimizer.step()
            scheduler.step()
            # gc.collect()
    return losses


device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
args = get_args()

tokenized_train_dataset = process_train_data("english")
train_dl = DataLoader(tokenized_train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=args.batch_size)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

lr = args.lr
n_epochs = args.epochs
weight_decay = 0.01
warmup_steps = 200

no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
     'weight_decay': weight_decay},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    warmup_steps,
    n_epochs * len(train_dl)
)

losses = train(
    model,
    train_dl,
    optimizer,
    scheduler,
    n_epochs,
    device
)

tokenized_val_dataset = process_validation_data("english")
example_val = load_dataset("copenlu/answerable_tydiqa")
example_val = getLanguageDataSet(example_val, "english")['validation']

val_dl = DataLoader(tokenized_train_dataset, collate_fn=val_collate_fn, shuffle=True, batch_size=args.batch_size)
logits = predict(model, val_dl)
predictions = post_process_predictions(example_val, tokenized_val_dataset, logits)
formatted_predictions = [{'id': k, 'prediction_text': v} for k, v in predictions.items()]
gold = [{'id': example['id'], 'answers': example['answers']} for example in example_val]