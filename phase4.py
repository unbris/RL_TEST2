import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import json
import random
from collections import deque

# ===========================
# 1. 配置参数
# ===========================
CONFIG = {
    'MAX_SEQ_LEN': 50,
    'EMBED_DIM': 64,
    'HIDDEN_DIM': 128,       # 知识状态向量的维度
    'MODEL_PATH': 'student_simulator.pth',
    'MAP_PATH': 'qid_map.json',
    'TEST_FILE': 'test.csv',
    'MAX_STEPS': 20,
    
    'RL_EPISODES': 1000,
    'RL_BATCH_SIZE': 64,
    'RL_LR': 0.0001,
    'GAMMA': 0.99,
    'EPSILON_START': 1.0,
    'EPSILON_END': 0.05,
    'EPSILON_DECAY': 0.997, 
    'MEMORY_SIZE': 50000
}

# ===========================
# 2. DKT 模型 (增加获取隐藏状态的功能)
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
        q_emb = self.q_embed(q_seq)
        r_reshaped = r_seq.unsqueeze(-1)
        input_feat = torch.cat([q_emb, r_reshaped], dim=-1)
        input_feat = self.interaction_embed(input_feat)
        lstm_out, _ = self.lstm(input_feat)
        logits = self.out(lstm_out)
        return logits

    # 🔥 新增功能：提取学生大脑状态 (Hidden State)
    def get_knowledge_state(self, q_seq, r_seq):
        q_emb = self.q_embed(q_seq)
        r_reshaped = r_seq.unsqueeze(-1)
        input_feat = torch.cat([q_emb, r_reshaped], dim=-1)
        input_feat = self.interaction_embed(input_feat)
        # 获取 LSTM 的最后时刻隐藏状态 (hn)
        _, (hn, cn) = self.lstm(input_feat)
        # hn shape: [num_layers, batch, hidden] -> 取最后一层: [batch, hidden]
        return hn[-1]

# ===========================
# 3. 环境 (输出 Advanced State)
# ===========================
class StudentEnv:
    def __init__(self, model_path, map_path, data_file):
        self.device = torch.device("cpu")
        with open(map_path, 'r') as f: self.qid_map = {int(k): v for k, v in json.load(f).items()}
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
            q_raw = [int(float(x)) for x in str(row['questions']).split(',') if x != 'nan' and x != '']
            r_raw = [int(float(x)) for x in str(row['responses']).split(',') if x != 'nan' and x != '']
            valid = [i for i, q in enumerate(q_raw) if q in self.qid_map]
            if len(valid) > 5: self.student_pool.append(([self.qid_map[q_raw[i]] for i in valid], [r_raw[i] for i in valid]))
        print(f"Env initialized. Action Space: 1 ~ {self.num_questions}")

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
        
        new_mastery = self._get_mastery()
        gain = new_mastery - self.history_mastery
        self.history_mastery = new_mastery
        
        # --- 混合奖励 (Hybrid Reward) ---
        reward = 0
        reward += gain * 500.0  # 稍微温和一点的放大，配合 ZPD
        
        # ZPD 引导 (回归初心，但有 Masking 保护，不怕死循环)
        if 0.4 <= pred_prob <= 0.7:
            reward += 0.5 
        
        # 惩罚极端
        if pred_prob > 0.95 or pred_prob < 0.1:
            reward -= 1.0
            
        # 严厉去重
        if self.current_q.count(action) > 1:
            reward = -5.0
            
        return self._get_state(), reward, self.step_count >= CONFIG['MAX_STEPS']

    def _get_mastery(self):
        with torch.no_grad():
            q_t = torch.tensor([self.current_q], dtype=torch.long).to(self.device)
            r_t = torch.tensor([self.current_r], dtype=torch.float).to(self.device)
            return torch.sigmoid(self.simulator(q_t, r_t)[0, -1, :]).mean().item()

    # 🔥 核心升级：返回知识向量 (Hidden State) 而不是 ID 序列
    def _get_state(self):
        q = self.current_q[-CONFIG['MAX_SEQ_LEN']:] + [0]*max(0, CONFIG['MAX_SEQ_LEN']-len(self.current_q))
        r = self.current_r[-CONFIG['MAX_SEQ_LEN']:] + [0]*max(0, CONFIG['MAX_SEQ_LEN']-len(self.current_r))
        q_t = torch.tensor([q[-CONFIG['MAX_SEQ_LEN']:]], dtype=torch.long).to(self.device)
        r_t = torch.tensor([r[-CONFIG['MAX_SEQ_LEN']:]], dtype=torch.float).to(self.device)
        
        with torch.no_grad():
            # 调用 DKT 的新方法获取 hidden state
            hidden_state = self.simulator.get_knowledge_state(q_t, r_t)
            # hidden_state shape: [1, 128] -> squeeze -> [128]
            return hidden_state.squeeze(0).numpy()

# ===========================
# 4. 新版 DQN (MLP 架构)
# ===========================
class DQN(nn.Module):
    def __init__(self, num_questions, state_dim, hidden_dim):
        super(DQN, self).__init__()
        # 输入不再是序列，而是知识向量 (state_dim = 128)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_questions + 1)

    def forward(self, state):
        # state shape: [batch, state_dim]
        x = self.relu(self.fc1(state))
        return self.fc2(x)

class Agent:
    def __init__(self, num_questions):
        self.device = torch.device("cpu")
        self.num_questions = num_questions
        # 注意：这里 DQN 的输入维度是 HIDDEN_DIM (128)
        self.policy_net = DQN(num_questions, CONFIG['HIDDEN_DIM'], 256).to(self.device)
        self.target_net = DQN(num_questions, CONFIG['HIDDEN_DIM'], 256).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=CONFIG['RL_LR'])
        self.memory = deque(maxlen=CONFIG['MEMORY_SIZE'])
        self.epsilon = CONFIG['EPSILON_START']

    def select_action(self, state, invalid_actions=None):
        if random.random() < self.epsilon:
            candidates = list(set(range(1, self.num_questions + 1)) - set(invalid_actions))
            return random.choice(candidates) if candidates else random.randint(1, self.num_questions)
        else:
            with torch.no_grad():
                state_t = torch.tensor([state], dtype=torch.float).to(self.device)
                q_vals = self.policy_net(state_t)
                if invalid_actions:
                    for idx in invalid_actions: 
                        if idx <= self.num_questions: q_vals[0, idx] = -1e9
                return q_vals[0, 1:].argmax().item() + 1

    def store_transition(self, s, a, r, ns, d): self.memory.append((s, a, r, ns, d))
    
    def update(self):
        if len(self.memory) < CONFIG['RL_BATCH_SIZE']: return
        batch = random.sample(self.memory, CONFIG['RL_BATCH_SIZE'])
        bs, ba, br, bns, bd = zip(*batch)
        
        # State 现在是向量，直接转 tensor
        state = torch.tensor(np.array(bs), dtype=torch.float).to(self.device)
        next_state = torch.tensor(np.array(bns), dtype=torch.float).to(self.device)
        actions = torch.tensor(ba, dtype=torch.long).unsqueeze(1).to(self.device)
        rewards = torch.tensor(br, dtype=torch.float).to(self.device)
        dones = torch.tensor(bd, dtype=torch.float).to(self.device)

        q_curr = self.policy_net(state).gather(1, actions).squeeze(1)
        with torch.no_grad():
            q_next = self.target_net(next_state)
            max_q_next = q_next[:, 1:].max(1)[0]
            target = rewards + (CONFIG['GAMMA'] * max_q_next * (1 - dones))
        
        loss = nn.MSELoss()(q_curr, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.epsilon > CONFIG['EPSILON_END']: self.epsilon *= CONFIG['EPSILON_DECAY']

# ===========================
# 5. 主训练循环 (SAVE BEST)
# ===========================
if __name__ == "__main__":
    print("Initializing Advanced RL System...")
    env = StudentEnv(CONFIG['MODEL_PATH'], CONFIG['MAP_PATH'], CONFIG['TEST_FILE'])
    agent = Agent(env.num_questions)
    
    print("Start Training (Advanced State + Hybrid Reward)...")
    history_rewards = []
    best_avg_reward = -float('inf') 
    
    for episode in range(CONFIG['RL_EPISODES']):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            invalid = list(env.current_q)
            action = agent.select_action(state, invalid)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward
        
        if episode % 20 == 0: agent.target_net.load_state_dict(agent.policy_net.state_dict())
        history_rewards.append(total_reward)
        
        avg = np.mean(history_rewards[-50:]) if len(history_rewards)>50 else np.mean(history_rewards)
        
        if episode > 100 and avg > best_avg_reward:
            best_avg_reward = avg
            torch.save(agent.policy_net.state_dict(), "rl_agent_advanced.pth") # 保存为新名字
            print(f"🌟 New Best: {best_avg_reward:.2f} (Ep {episode+1})")
        
        if (episode+1) % 50 == 0:
            print(f"Ep {episode+1} | Total: {total_reward:.2f} | Avg: {avg:.2f} | Best: {best_avg_reward:.2f}")

    print(f"Done! Best Model saved as 'rl_agent_advanced.pth'")