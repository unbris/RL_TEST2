import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import json 


CONFIG = {
    'MAX_SEQ_LEN': 100,
    'EMBED_DIM': 64,
    'HIDDEN_DIM': 128,
    'BATCH_SIZE': 32,
    'EPOCHS': 10,
    'LR': 0.001,
    'TRAIN_FILE': 'train.xlsx', 
    'TEST_FILE': 'test.csv'
}


def build_and_save_map(file_paths):
    print("Scanning files to build ID map...")
    unique_q = set()
    
    for fpath in file_paths:
        try:
            if fpath.endswith('.xlsx'):
                df = pd.read_excel(fpath, engine='openpyxl')
            else:
                df = pd.read_csv(fpath, encoding='utf-8-sig', engine='python', on_bad_lines='skip')
        except:
            print(f"Skipping {fpath} due to load error.")
            continue
            
        for q_raw in df['questions']:
            if pd.isna(q_raw): continue
           
            q_ids = [int(float(x)) for x in str(q_raw).split(',') if x != '']
            unique_q.update(q_ids)
            

    sorted_ids = sorted(list(unique_q))
    qid_map = {original_id: idx+1 for idx, original_id in enumerate(sorted_ids)}
    
    print(f"Found {len(qid_map)} unique questions. Map saved.")
    

    with open('qid_map.json', 'w') as f:
        json.dump(qid_map, f)
        
    return qid_map


class KTDataset(Dataset):
    def __init__(self, file_path, qid_map, max_len=100):
        self.max_len = max_len
        self.samples = []
        self.qid_map = qid_map 
        
        print(f"Processing {file_path}...")
        if file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path, engine='openpyxl')
        else:
            try:
                self.df = pd.read_csv(file_path, encoding='utf-8-sig', engine='python', on_bad_lines='skip')
            except:
                self.df = pd.read_csv(file_path, encoding='gbk', engine='python', on_bad_lines='skip')

        for idx, row in self.df.iterrows():
            q_raw = row['questions']
            r_raw = row['responses']
            if pd.isna(q_raw) or pd.isna(r_raw): continue
            
            q_seq_raw = [int(float(x)) for x in str(q_raw).split(',') if x != '']
            r_seq = [int(float(x)) for x in str(r_raw).split(',') if x != '']
            
            
            q_seq_mapped = [self.qid_map[q] for q in q_seq_raw if q in self.qid_map]
            
            
            valid_indices = [i for i, q in enumerate(q_seq_raw) if q in self.qid_map]
            q_seq = [self.qid_map[q_seq_raw[i]] for i in valid_indices]
            r_seq = [r_seq[i] for i in valid_indices]
            
            min_len = min(len(q_seq), len(r_seq))
            if min_len == 0: continue
            
            q_seq = q_seq[:min_len][-max_len:] 
            r_seq = r_seq[:min_len][-max_len:]
            
            self.samples.append((q_seq, r_seq))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        q_seq, r_seq = self.samples[idx]
        return torch.tensor(q_seq, dtype=torch.long), torch.tensor(r_seq, dtype=torch.float)

def collate_fn(batch):
    q_batch, r_batch = zip(*batch)
    q_padded = pad_sequence(q_batch, batch_first=True, padding_value=0)
    r_padded = pad_sequence(r_batch, batch_first=True, padding_value=-1)
    return q_padded, r_padded


class DKT(nn.Module):
    def __init__(self, num_questions, embed_dim, hidden_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.q_embed = nn.Embedding(num_questions + 1, embed_dim, padding_idx=0)
        self.interaction_embed = nn.Linear(embed_dim + 1, embed_dim) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_questions + 1)

    def forward(self, q_seq, r_seq):
        q_emb = self.q_embed(q_seq)
        r_reshaped = r_seq.unsqueeze(-1)
        input_feat = torch.cat([q_emb, r_reshaped], dim=-1)
        input_feat = self.interaction_embed(input_feat)
        lstm_out, _ = self.lstm(input_feat)
        logits = self.out(lstm_out)
        return logits


def train():
   
    qid_map = build_and_save_map([CONFIG['TRAIN_FILE'], CONFIG['TEST_FILE']])
    
   
    NUM_QUESTIONS_MAPPED = len(qid_map)
    print(f"Total Mapped Questions: {NUM_QUESTIONS_MAPPED}")
    
   
    train_dataset = KTDataset(CONFIG['TRAIN_FILE'], qid_map, CONFIG['MAX_SEQ_LEN'])
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn)
    
    
    device = torch.device("cpu") 
    print("Using CPU for training...")
    
    model = DKT(NUM_QUESTIONS_MAPPED, CONFIG['EMBED_DIM'], CONFIG['HIDDEN_DIM']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['LR'])
    criterion = nn.BCEWithLogitsLoss()

    print("Start Training...")
    for epoch in range(CONFIG['EPOCHS']):
        model.train()
        total_loss = 0
        for q_batch, r_batch in train_loader:
            q_batch, r_batch = q_batch.to(device), r_batch.to(device)
            optimizer.zero_grad()
            
            in_q = q_batch[:, :-1]
            in_r = r_batch[:, :-1]
            target_r = r_batch[:, 1:]
            target_q = q_batch[:, 1:] 
            
            if in_q.shape[1] == 0: continue
            
            logits = model(in_q, in_r)
            target_q_idx = target_q.unsqueeze(-1)
            pred_logits = torch.gather(logits, 2, target_q_idx).squeeze(-1)
            
            mask = (target_r != -1)
            if mask.sum() == 0: continue 
            loss = criterion(pred_logits[mask], target_r[mask])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")
    
    torch.save(model.state_dict(), "student_simulator.pth")
    print("Simulator Model & Map Saved!")

if __name__ == "__main__":
    train()