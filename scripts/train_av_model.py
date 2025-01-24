import torch
from torch.utils.data import DataLoader
from models.av_model import AVBiLSTMModel
from data.dataset import TVSumDataset, SumMeDataset


def train(config):
    # Initialize datasets
    tvsum_train = TVSumDataset(
        "Evaluation/TVSum/ydata-tvsum50-data/data/ydata-tvsum50-anno.tsv", "features"
    )

    summe_train = SumMeDataset("Evaluation/SumMe/GT", "features")

    # Combined dataset
    full_dataset = torch.utils.data.ConcatDataset([tvsum_train, summe_train])
    train_loader = DataLoader(
        full_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn
    )

    # Model and optimizer
    model = AVBiLSTMModel().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(100):
        model.train()
        for batch in train_loader:
            visual, audio, scores = batch
            outputs = model(visual.cuda(), audio.cuda())
            loss = criterion(outputs.squeeze(), scores.cuda())

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Validation
        val_loss = validate(model, val_loader)
        print(
            f"Epoch {epoch+1} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}"
        )


def collate_fn(batch):
    visual = [item[0]["visual"] for item in batch]
    audio = [item[0]["audio"] for item in batch]
    scores = [item[1] for item in batch]

    # Pad sequences to same length
    visual = torch.nn.utils.rnn.pad_sequence(visual, batch_first=True)
    audio = torch.nn.utils.rnn.pad_sequence(audio, batch_first=True)
    scores = torch.nn.utils.rnn.pad_sequence(scores, batch_first=True)

    return visual, audio, scores
