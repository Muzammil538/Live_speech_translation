import os
import torch
import torchaudio
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class SpeechDataset(Dataset):
    def __init__(self, data_dir, labels):
        self.data = []
        self.labels = labels
        for fname in os.listdir(data_dir):
            if fname.endswith(".wav"):
                self.data.append(os.path.join(data_dir, fname))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.data[idx])
        mfcc = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=13)(waveform).squeeze(0).transpose(0, 1)
        label = self.labels[os.path.basename(self.data[idx]).split(".")[0]]
        return mfcc, torch.tensor([ord(c)-96 for c in label.lower() if c.isalpha()])

class CTCModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x.log_softmax(2)

def collate_fn(batch):
    xs, ys = zip(*batch)
    x_lens = torch.tensor([x.size(0) for x in xs])
    y_lens = torch.tensor([y.size(0) for y in ys])
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True)
    ys = torch.cat(ys)
    return xs, ys, x_lens, y_lens

data_dir = "vosk-model-small-en-in-0.4"
labels = {"sample1":"hello","sample2":"world"}  # replace with real labels
dataset = SpeechDataset(data_dir, labels)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CTCModel(13, 256, 27).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(300):
    total_loss=0
    for x,y,xlen,ylen in loader:
        x,y=x.to(device),y.to(device)
        out=model(x)
        out=out.permute(1,0,2)
        loss=criterion(out,y,xlen,ylen)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1}/300 Loss: {total_loss/len(loader):.4f}")
os.makedirs("vosk_model", exist_ok=True)
torch.save(model.state_dict(),"vosk_model/model.pt")
print("Model saved to vosk_model/model.pt")