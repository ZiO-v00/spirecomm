import json
import os

def extract_floor_transitions(log_path, output_path):
    transitions = []
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    prev_state = None
    prev_hp = None
    for line in lines:
        data = json.loads(line)
        if data.get("_type", "").startswith("state:floor"):
            # 報酬計算用に前回のhp_currentを保存
            if prev_state is not None and prev_hp is not None:
                curr_hp = data.get("hp_current", None)
                reward = 0
                if curr_hp is not None:
                    reward = curr_hp - prev_hp  # HP増減を報酬に
                transitions[-1]["reward"] = reward
            prev_state = data
            prev_hp = data.get("hp_current", None)
        elif data.get("_type", "").startswith("action:"):
            action = data
            transitions.append({
                "floor": prev_state.get("floor") if prev_state else None,
                "state": prev_state,
                "action": action,
                "reward": None  # 次のstate:floorで計算
            })
    # 保存（例：JSON形式）
    with open(output_path, 'w', encoding='utf-8') as out:
        json.dump(transitions, out, ensure_ascii=False, indent=2)

# 使い方例
if __name__ == "__main__":
    # logファイルのパスと出力先を指定
    log_path = "../../Mods/spirecomm/runs/sample.run.log"  # 実際のパスに合わせて修正
    output_path = "floor_transitions.json"
    extract_floor_transitions(log_path, output_path)
