import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import json
import random

# ===========================
# 配置
# ===========================
CONFIG = {
    'MAX_SEQ_LEN': 50,
    'EMBED_DIM': 64,
    'HIDDEN_DIM': 128,
    'MODEL_PATH': 'student_simulator.pth',
    'MAP_PATH': 'qid_map.json',
    'TEST_FILE': 'test.csv', 
    'MAX_STEPS': 20
}

# ===========================
# 模型定义
# ===========================
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

# ===========================
# 环境定义 (同步 Phase 4 的新逻辑)
# ===========================
class StudentEnv:
    def __init__(self, model_path, map_path, data_file):
        self.device = torch.device("cpu")
        with open(map_path, 'r') as f: self.qid_map = {int(k): v for k, v in json.load(f).items()}
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
            valid = [i for i, q in enumerate(q_raw) if q in self.qid_map]
            if len(valid) > 5: 
                self.student_pool.append(([self.qid_map[q_raw[i]] for i in valid], [r_raw[i] for i in valid]))
        print(f"Environment initialized with {len(self.student_pool)} students.")

    def reset(self):
        self.current_q, self.current_r = random.choice(self.student_pool)
        if len(self.current_q) > 20: start=0; self.current_q=self.current_q[:10]; self.current_r=self.current_r[:10]
        else: self.current_q=list(self.current_q); self.current_r=list(self.current_r)
        self.step_count = 0; self.history_mastery = self._get_mastery()
        return self._get_state()

    def step(self, action):
        self.step_count += 1
        q_t = torch.tensor([self.current_q], dtype=torch.long).to(self.device)
        r_t = torch.tensor([self.current_r], dtype=torch.float).to(self.device)
        with torch.no_grad():
            logits = self.simulator(q_t, r_t)
            pred_prob = torch.sigmoid(logits[0, -1, action]).item()
        
        response = 1 if random.random() < pred_prob else 0
        self.current_q.append(action); self.current_r.append(response)
        
        new_mastery = self._get_mastery(); gain = new_mastery - self.history_mastery
        self.history_mastery = new_mastery
        
        # --- 同步的新奖励函数 ---
        reward = 0
        reward += gain * 200.0 
        
        # 难度奖励 (ZPD)
        if 0.3 <= pred_prob <= 0.75: reward += 1.5
        elif pred_prob > 0.9: reward -= 1.0
        elif pred_prob < 0.2: reward -= 1.0
            
        # 重复惩罚
        if self.current_q.count(action) > 1: reward = -5.0
            
        original_qid = self.idx_to_qid.get(action, -1)
        done = self.step_count >= CONFIG['MAX_STEPS']
        
        return self._get_state(), reward, done, {
            'prob': pred_prob, 'response': response, 'mastery': new_mastery, 'original_id': original_qid
        }

    def _get_mastery(self):
        with torch.no_grad():
            q_t = torch.tensor([self.current_q], dtype=torch.long).to(self.device)
            r_t = torch.tensor([self.current_r], dtype=torch.float).to(self.device)
            return torch.sigmoid(self.simulator(q_t, r_t)[0, -1, :]).mean().item()

    def _get_state(self):
        q = self.current_q[-CONFIG['MAX_SEQ_LEN']:] + [0]*max(0, CONFIG['MAX_SEQ_LEN']-len(self.current_q))
        r = self.current_r[-CONFIG['MAX_SEQ_LEN']:] + [0]*max(0, CONFIG['MAX_SEQ_LEN']-len(self.current_r))
        return np.array(q[-CONFIG['MAX_SEQ_LEN']:]), np.array(r[-CONFIG['MAX_SEQ_LEN']:])

# ===========================
# 测试运行
# ===========================
if __name__ == "__main__":
    env = StudentEnv(CONFIG['MODEL_PATH'], CONFIG['MAP_PATH'], CONFIG['TEST_FILE'])
    state = env.reset()
    print("\n--- Starting Simulation (Random Agent) ---")
    valid_actions = list(env.qid_map.values())
    
    for t in range(5):
        action = random.choice(valid_actions) # 随机动作
        next_state, reward, done, info = env.step(action)
        print(f"Step {t+1}: Rec QID {info['original_id']} | Prob: {info['prob']:.1%} | Reward: {reward:.2f}")