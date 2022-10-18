# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/


lstm_dim = 300
dropout_prob = 0.1
batch_size = 64
lr = 1e-3
n_epochs = 20

device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")

# Create the model
finnish_model = Seq2Seq(
    pretrained_embeddings=torch.FloatTensor(finnish_pretrained_embeddings),
    lstm_dim=lstm_dim,
    tokenizer = finnish_tokenizer,
    dropout_prob=dropout_prob,
    n_classes=3
  ).to(device)


finnish_train_dl = DataLoader(finnish_tokenized_datasets['train'], batch_size=batch_size, shuffle=True, collate_fn=finnish_collate_batch_bilstm, num_workers=8)
finnish_valid_dl = DataLoader(finnish_tokenized_datasets['validation'], batch_size=1, collate_fn=finnish_collate_batch_bilstm, num_workers=8)

# Create the optimizer
finnish_optimizer = Adam(finnish_model.parameters(), lr=lr)

# Train
losses = train(finnish_model, finnish_train_dl, finnish_valid_dl, finnish_optimizer, n_epochs, device, finnish_tokenizer)
finnish_model.load_state_dict(torch.load('best_model'))