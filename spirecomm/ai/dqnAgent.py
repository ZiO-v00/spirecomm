import json
import time
import os
import re
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from spirecomm.ai.agent import Agent
from spirecomm.spire.character import PlayerClass
from collections import deque
from spirecomm.communication.action import PlayCardAction, EndTurnAction
from spirecomm.communication.action import StartGameAction

class QNetwork(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(QNetwork, self).__init__()
		self.fc1 = nn.Linear(state_dim, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, action_dim)

	def forward(self, x):
		x = torch.relu(self.fc1(x))
		x = torch.relu(self.fc2(x))
		return self.fc3(x)

class DqnAgent(Agent):

	# ===== DQN独自の学習・推論メソッド =====

	def learn_new_floor_from_runlog(self, runlog_path, last_learned_floor):
		"""
		run.logをパースし、last_learned_floorより新しいfloorが出現したら、
		その直前のfloorの遷移データだけを学習し、新しいfloor番号を返す。
		新しいfloorがなければlast_learned_floorを返す。
		"""
		import json
		try:
			with open(runlog_path, 'r', encoding='utf-8') as f:
				lines = f.readlines()
		except Exception as e:
			print(f"run.log読み込み失敗: {e}")
			return last_learned_floor
		states = []
		actions = []
		for line in lines:
			data = json.loads(line)
			if data.get('_type', '').startswith('state:floor'):
				states.append(data)
			elif data.get('_type', '').startswith('action:'):
				actions.append(data)
		# 新しいfloorが出現したか判定
		if len(states) >= 2:
			prev_floor = states[-2].get('floor', -1)
			curr_floor = states[-1].get('floor', -1)
			if prev_floor > last_learned_floor:
				# prev_floorのactionを抽出
				floor_actions = [a for a in actions if a.get('floor') == prev_floor]
				ns = states[-1]
				s = states[-2]
				for action in floor_actions:
					reward = ns.get('hp_current', 0) - s.get('hp_current', 0)
					transition = {
						'state': s,
						'action': action,
						'next_state': ns,
						'reward': reward
					}
					self.train_from_transitions([transition])
				return prev_floor
		return last_learned_floor

	def train_from_transitions(self, transitions):
		"""
		transitionsデータを使ってDQNの学習を行う（雛形）
		"""
		# 実装例: pass
		pass

	# ===== 継承（super）で親クラスの自動進行ロジックを使うメソッド =====

	def get_next_combat_action(self):
		# DQNによる手札選択ロジックをここに入れる場合はこの位置で。
		# 何もなければ親クラスの自動進行ロジックを使う。
		return super().get_next_combat_action()

	def get_card_reward_action(self):
		return super().get_card_reward_action()

	def get_rest_action(self):
		return super().get_rest_action()

	def get_screen_action(self, screen_type=None, screen_state=None):
		print(f"[DqnAgent] get_screen_action called: screen_type={screen_type}, screen_state={screen_state}")
		return super().get_screen_action(screen_type, screen_state)

	def get_map_choice_action(self):
		return super().get_map_choice_action()

	def get_next_combat_reward_action(self):
		return super().get_next_combat_reward_action()

	def get_next_boss_reward_action(self):
		return super().get_next_boss_reward_action()
	def get_next_combat_action(self):
		# DQNによる手札選択ロジックをここに入れる場合はこの位置で。
		# 何もなければ親クラスの自動進行ロジックを使う。
		return super().get_next_combat_action()

	def get_card_reward_action(self):
		return super().get_card_reward_action()

	def get_rest_action(self):
		return super().get_rest_action()

	def get_screen_action(self, screen_type=None, screen_state=None):
		print("called", screen_type, screen_state)
		return super().get_screen_action(screen_type, screen_state)

	def get_map_choice_action(self):
		return super().get_map_choice_action()

	def get_next_combat_reward_action(self):
		return super().get_next_combat_reward_action()

	def get_next_boss_reward_action(self):
		return super().get_next_boss_reward_action()
	def get_next_combat_action(self):
		# ここにDQNによる手札選択ロジックを実装（例: index=0固定など）
		# 必要に応じて状態からアクションを決定
		# 例: return PlayCardAction(card_index=0)
		# 手札がなければターン終了
		if not self.game.hand:
			return EndTurnAction()
		# 例: 1枚目のカードを常に選択（DQNロジックに置き換え可）
		return PlayCardAction(card_index=0)
	def learn_new_floor_from_runlog(self, runlog_path, last_learned_floor):
		"""
		run.logをパースし、last_learned_floorより新しいfloorが出現したら、
		その直前のfloorの遷移データだけを学習し、新しいfloor番号を返す。
		新しいfloorがなければlast_learned_floorを返す。
		"""
		try:
			with open(runlog_path, 'r', encoding='utf-8') as f:
				lines = f.readlines()
		except Exception as e:
			print(f"run.log読み込み失敗: {e}")
			return last_learned_floor
		states = []
		actions = []
		for line in lines:
			data = json.loads(line)
			if data.get('_type', '').startswith('state:floor'):
				states.append(data)
			elif data.get('_type', '').startswith('action:'):
				actions.append(data)
		# 新しいfloorが出現したか判定
		if len(states) >= 2:
			prev_floor = states[-2].get('floor', -1)
			curr_floor = states[-1].get('floor', -1)
			if prev_floor > last_learned_floor:
				# prev_floorのactionを抽出
				floor_actions = [a for a in actions if a.get('floor') == prev_floor]
				ns = states[-1]
				s = states[-2]
				for action in floor_actions:
					reward = ns.get('hp_current', 0) - s.get('hp_current', 0)
					transition = {
						'state': s,
						'action': action,
						'next_state': ns,
						'reward': reward
					}
					self.train_from_transitions([transition])
				return prev_floor
		return last_learned_floor

	def __init__(self):
		super().__init__()
		# ...既存の初期化処理...
		self.last_learned_floor = -1

	def learn_from_runlog(self, runlog_path):
		"""
		run.logをパースし、新規フロアのみ状態・行動・報酬遷移を抽出して学習する
		"""
		import json
		try:
			with open(runlog_path, 'r', encoding='utf-8') as f:
				lines = f.readlines()
		except Exception as e:
			print(f"run.log読み込み失敗: {e}")
			return
		states = []
		actions = []
		for line in lines:
			data = json.loads(line)
			if data.get('_type', '').startswith('state:floor'):
				states.append(data)
			elif data.get('_type', '').startswith('action:'):
				actions.append(data)
		# 新規フロアのみ抽出
		transitions = []
		for i in range(min(len(states)-1, len(actions))):
			s = states[i]
			ns = states[i+1]
			floor_num = s.get('floor', -1)
			if floor_num > self.last_learned_floor:
				reward = ns.get('hp_current', 0) - s.get('hp_current', 0)
				transition = {
					'state': s,
					'action': actions[i],
					'next_state': ns,
					'reward': reward
				}
				transitions.append(transition)
		if transitions:
			self.train_from_transitions(transitions)
			# 最後に学習したフロア番号を更新
			self.last_learned_floor = max([t['state'].get('floor', -1) for t in transitions])
		else:
			print("run.logから新規学習データが抽出できませんでした")

	def auto_learn_loop(self, runlog_dir, extract_script_path, output_path, interval=60):
		"""
		指定ディレクトリの最新.run.logを定期的に抽出・変換・学習する自動ループ
		runlog_dir: .run.logファイルのディレクトリ
		extract_script_path: extract_floor_transitions.pyのパス
		output_path: 変換後データの保存先
		interval: チェック間隔（秒）
		"""
		last_max_num = None
		while True:
			files = [f for f in os.listdir(runlog_dir) if f.endswith('.run.log')]
			nums = [int(f.split('.')[0]) for f in files if f.split('.')[0].isdigit()]
			if not nums:
				print("No .run.log files found.")
				time.sleep(interval)
				continue
			max_num = max(nums)
			if max_num != last_max_num:
				latest_log = os.path.join(runlog_dir, f"{max_num}.run.log")
				# 抽出スクリプトを実行
				os.system(f'python "{extract_script_path}" "{latest_log}" "{output_path}"')
				# 学習データを読み込み・学習
				self.load_transitions(output_path)
				self.train_from_transitions()
				last_max_num = max_num
				print(f"Learned from {latest_log}")
			time.sleep(interval)
	def load_latest_runlog(self, runlog_dir):
		"""
		指定ディレクトリから最も値が大きい.run.logファイルを自動選択して読み込む
		"""
		files = [f for f in os.listdir(runlog_dir) if f.endswith('.run.log')]
		# ファイル名が数字のみのものを抽出
		num_files = []
		for f in files:
			match = re.match(r'^(\d+)\.run\.log$', f)
			if match:
				num_files.append((int(match.group(1)), f))
		if not num_files:
			print("No numeric .run.log files found.")
			return
		# 最大値のファイルを選択
		max_file = max(num_files, key=lambda x: x[0])[1]
		path = os.path.join(runlog_dir, max_file)
		print(f"Loading latest runlog: {path}")
		self.load_transitions(path)
	def __init__(self):
		super().__init__()
		self.state_dim = 9
		self.action_dim = 5
		self.q_network = QNetwork(self.state_dim, self.action_dim)
		self.target_network = QNetwork(self.state_dim, self.action_dim)
		self.target_network.load_state_dict(self.q_network.state_dict())
		self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
		self.replay_buffer = deque(maxlen=10000)
		self.batch_size = 32
		self.gamma = 0.99
		self.epsilon = 0.1
		self.update_target_steps = 100
		self.learn_step = 0
		self.transitions = []
		runlog_dir = r"C:\Program Files (x86)\Steam\steamapps\common\SlayTheSpire\runlogs\DEFECT"
		try:
			self.load_latest_runlog(runlog_dir)
		except Exception:
			self.transitions = []
	def preprocess_state(self, state):
		# Slay the Spire用の状態ベクトル化
		hp_current = state.get("hp_current", 0)
		hp_max = state.get("hp_max", 0)
		gold = state.get("gold", 0)
		deck_size = len(state.get("deck", []))
		potions = sum([1 for p in state.get("potions", []) if p is not None])
		relics = len(state.get("relics", []))
		floor = state.get("floor", 0)
		# 敵情報
		enemy_hp = 0
		enemy_intent = 0
		combat = state.get("combat_state", {})
		monsters = combat.get("monsters", [])
		if monsters:
			enemy_hp = monsters[0].get("hp_current", 0)
			intent_map = {"ATTACK": 1, "BUFF": 2, "DEBUFF": 3}
			enemy_intent = intent_map.get(monsters[0].get("intent", ""), 0)
		return np.array([
			hp_current, hp_max, gold, deck_size, potions, relics, floor, enemy_hp, enemy_intent
		], dtype=np.float32)

	def preprocess_action(self, action):
		# Slay the Spire用の行動インデックス化
		if action.get("_type") == "action:play_card":
			card = action.get("card", "")
			card_map = {"Strike": 0, "Defend": 1, "Zap": 2, "Dualcast": 3}
			return card_map.get(card, 4)  # 未知カードはEndTurn扱い
		elif action.get("_type") == "action:end_turn":
			return 4
		return 4

	def get_next_action_in_game(self, state):
		"""
		戦闘中（手札がある場合）のみDQNで行動決定し、それ以外はindex=0を選択
		"""
		combat = state.get("combat_state", {})
		hand = combat.get("hand", [])
		if hand:
			# DQNで手札選択
			state_vec = self.preprocess_state(state)
			state_tensor = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)
			if np.random.rand() < self.epsilon:
				action_idx = np.random.randint(self.action_dim)
			else:
				with torch.no_grad():
					q_values = self.q_network(state_tensor)
					action_idx = torch.argmax(q_values).item()
			return self.action_index_to_game_action(action_idx, state)
		# 戦闘中でなければ（optionsがある画面等）はindex=0を選択
		screen_state = state.get("screen_state", {})
		options = screen_state.get("options", [])
		if options:
			return {"_type": "action:select_dialog", "index": 0}
		# それ以外はend_turn
		return {"_type": "action:end_turn"}

	def action_index_to_game_action(self, index, state):
		# インデックスからカード名や行動種別を決定
		card_list = ["Strike", "Defend", "Zap", "Dualcast"]
		if index < len(card_list):
			# 手札にあればそのカードを使う
			hand = state.get("combat_state", {}).get("hand", [])
			for i, card in enumerate(hand):
				if card == card_list[index]:
					return {"_type": "action:play_card", "card": card, "card_index": i}
			# なければEndTurn
			return {"_type": "action:end_turn"}
		else:
			return {"_type": "action:end_turn"}

	def train_from_transitions(self):
		if not hasattr(self, 'transitions'):
			print("No transitions loaded.")
			return
		# transitionsからリプレイバッファに追加
		for t in self.transitions:
			state = self.preprocess_state(t.get("state"))
			action = self.preprocess_action(t.get("action"))
			reward = t.get("reward", 0)
			next_state = self.preprocess_state(t.get("next_state", t.get("state")))
			done = False  # 必要に応じて終了判定
			self.replay_buffer.append((state, action, reward, next_state, done))
		# バッチ学習
		if len(self.replay_buffer) < self.batch_size:
			return
		batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
		states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[idx] for idx in batch])
		states = torch.tensor(states, dtype=torch.float32)
		actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
		rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
		next_states = torch.tensor(next_states, dtype=torch.float32)
		dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
		# Q値計算
		q_values = self.q_network(states).gather(1, actions)
		with torch.no_grad():
			target_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
			target = rewards + self.gamma * target_q * (1 - dones)
		loss = nn.functional.mse_loss(q_values, target)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
		self.learn_step += 1
		if self.learn_step % self.update_target_steps == 0:
			self.target_network.load_state_dict(self.q_network.state_dict())

	def handle_error(self, error):
		# エラー処理
		pass


	def get_next_action_out_of_game(self):
		# ゲーム外（イベントやショップ等）の行動選択
		return StartGameAction(self.chosen_class)

	def change_class(self, chosen_class: PlayerClass):
		# プレイヤークラスの変更
		self.chosen_class = chosen_class
	
	def load_transitions(self, path):
		"""
		階層ごとの状態・行動・報酬データを読み込む
		"""
		with open(path, 'r', encoding='utf-8') as f:
			self.transitions = json.load(f)

	def train_from_transitions(self):
		"""
		transitionsデータを使ってDQNの学習を行う（雛形）
		"""
		if not hasattr(self, 'transitions'):
			print("No transitions loaded.")
			return
		for t in self.transitions:
			state = t.get("state")
			action = t.get("action")
			reward = t.get("reward")
			# ここでDQNの学習処理を実装
			# 例: 状態・行動・報酬をネットワークに入力
			pass
