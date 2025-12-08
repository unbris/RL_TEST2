import torch
import numpy as np
import pandas as pd
import json
import random
import torch.nn as nn


CONFIG = {
    'MAX_SEQ_LEN': 50,
    'EMBED_DIM': 64,
    'HIDDEN_DIM': 128,
    'MODEL_PATH': 'student_simulator.pth',
    'MAP_PATH': 'qid_map.json',
    'AGENT_PATH': 'rl_agent.pth', 
    'TEST_FILE': 'test.csv',
}


class DKT(nn.Module):
    def __init__(self, num_questions, embed_dim, hidden_dim):
        super(DKT, self).__init__()
        self.hidden_dim = hidden_dim
        self.q_embed = nn.Embedding(num_questions + 1, embed_dim, padding_idx=0)
        self.interaction_embed = nn.Linear(embed_dim + 1, embed_dim) 
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_questions + 1)
    def forward(self, q_seq, r_seq):
        q_emb = self.q_embed(q_seq); r_reshaped = r_seq.unsqueeze(-1)
        input_feat = torch.cat([q_emb, r_reshaped], dim=-1) 
        input_feat = self.interaction_embed(input_feat)     
        lstm_out, _ = self.lstm(input_feat); return self.out(lstm_out)

class DQN(nn.Module):
    def __init__(self, num_questions, embed_dim, hidden_dim):
        super(DQN, self).__init__()
        self.q_embed = nn.Embedding(num_questions + 1, embed_dim, padding_idx=0)
        self.interaction_embed = nn.Linear(embed_dim + 1, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, num_questions + 1)
    def forward(self, q_seq, r_seq):
        q_emb = self.q_embed(q_seq); r_reshaped = r_seq.unsqueeze(-1)
        x = torch.cat([q_emb, r_reshaped], dim=-1); x = self.interaction_embed(x)
        lstm_out, _ = self.lstm(x); return self.out(lstm_out[:, -1, :])

class StudentEnv:
    def __init__(self, model_path, map_path, data_file):
        self.device = torch.device("cpu")
        with open(map_path, 'r') as f:
            self.qid_map = {int(k): v for k, v in json.load(f).items()}
        self.idx_to_qid = {v: k for k, v in self.qid_map.items()}
        self.num_questions = len(self.qid_map)
        self.simulator = DKT(self.num_questions, CONFIG['EMBED_DIM'], CONFIG['HIDDEN_DIM'])
        try: self.simulator.load_state_dict(torch.load(model_path, map_location=self.device))
        except: self.simulator.load_state_dict(torch.load(model_path, map_location=self.device), strict=False)
        self.simulator.to(self.device); self.simulator.eval()
        self.student_pool = []
        try:
            if data_file.endswith('.xlsx'): df = pd.read_excel(data_file, engine='openpyxl')
            else: df = pd.read_csv(data_file, encoding='utf-8-sig', engine='python', on_bad_lines='skip')
        except: df = pd.read_csv(data_file, encoding='gbk', engine='python', on_bad_lines='skip')
        for _, row in df.iterrows():
            q_str, r_str = str(row['questions']), str(row['responses'])
            if 'nan' in q_str.lower(): continue
            q_raw = [int(float(x)) for x in q_str.split(',') if x != '']
            r_raw = [int(float(x)) for x in r_str.split(',') if x != '']
            valid_indices = [i for i, q in enumerate(q_raw) if q in self.qid_map]
            q_mapped = [self.qid_map[q_raw[i]] for i in valid_indices]
            r_mapped = [r_raw[i] for i in valid_indices]
            if len(q_mapped) > 5: self.student_pool.append((q_mapped, r_mapped))

    def reset(self):
        self.current_q, self.current_r = random.choice(self.student_pool)
        if len(self.current_q) > 20: start=0; self.current_q=self.current_q[:10]; self.current_r=self.current_r[:10]
        else: self.current_q=list(self.current_q); self.current_r=list(self.current_r)
        self.step_count = 0; return self._get_state()

    def step(self, action):
        self.step_count += 1
        q_t = torch.tensor([self.current_q], dtype=torch.long).to(self.device)
        r_t = torch.tensor([self.current_r], dtype=torch.float).to(self.device)
        with torch.no_grad():
            logits = self.simulator(q_t, r_t)
            pred_prob = torch.sigmoid(logits[0, -1, action]).item()
        response = 1 if random.random() < pred_prob else 0
        self.current_q.append(action); self.current_r.append(response)
        original_qid = self.idx_to_qid.get(action, "Unknown")
        return self._get_state(), response, pred_prob, original_qid

    def _get_state(self):
        q = self.current_q[-CONFIG['MAX_SEQ_LEN']:] + [0]*max(0, CONFIG['MAX_SEQ_LEN']-len(self.current_q))
        r = self.current_r[-CONFIG['MAX_SEQ_LEN']:] + [0]*max(0, CONFIG['MAX_SEQ_LEN']-len(self.current_r))
        return np.array(q[-CONFIG['MAX_SEQ_LEN']:]), np.array(r[-CONFIG['MAX_SEQ_LEN']:])


def demo():
    print("Loading Environment and Agent...")
    env = StudentEnv(CONFIG['MODEL_PATH'], CONFIG['MAP_PATH'], CONFIG['TEST_FILE'])
    agent_net = DQN(env.num_questions, CONFIG['EMBED_DIM'], CONFIG['HIDDEN_DIM'])
    try: agent_net.load_state_dict(torch.load(CONFIG['AGENT_PATH'], map_location='cpu'))
    except: agent_net.load_state_dict(torch.load(CONFIG['AGENT_PATH'], map_location='cpu'), strict=False)
    agent_net.eval()
    
    print("\n" + "="*50)
    print("🎓 Personalized Learning Recommendation DEMO (Optimized)")
    print("="*50)
    
    state = env.reset()
    print(f"Student Initial History: {len(env.current_q)} questions done.")
    print("-" * 50)
    
    for t in range(10): 
        state_q, state_r = state
        
        
        already_done = set(env.current_q)
        
        with torch.no_grad():
            q_vals = agent_net(torch.tensor([state_q], dtype=torch.long), torch.tensor([state_r], dtype=torch.float))
            
           
            for idx in already_done:
                if idx < q_vals.shape[1]: q_vals[0, idx] = -1e9
            q_vals[0, 0] = -1e9 # Mask padding
            
            action = q_vals[0, 1:].argmax().item() + 1
            
        next_state, response, prob, original_qid = env.step(action)
        state = next_state
        
        status = "✅ Correct" if response == 1 else "❌ Incorrect"
        print(f"Step {t+1}: Agent Recommends QID {original_qid} (Mapped {action})")
        print(f"        -> Simulator Prediction: {prob:.1%} chance to solve")
        print(f"        -> Student Result: {status}")
        print("-" * 50)

if __name__ == "__main__":
    demo()