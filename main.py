
import os
import sys
import time
import random
import logging
import threading

from spirecomm.communication.action import PlayCardAction, StartGameAction
from spirecomm.communication.coordinator import Coordinator
from spirecomm.ai.simpleAgent import SimpleAgent
from spirecomm.ai.nnAgent import NnAgent
from spirecomm.ai.dqnAgent import DqnAgent
from spirecomm.ai.agent import Agent
from spirecomm.spire.character import PlayerClass


def main():
	logging.basicConfig(filename='neuralNet.log', level=logging.DEBUG)
	agent: Agent = DqnAgent()
	coordinator = Coordinator()
	coordinator.signal_ready()
	coordinator.register_command_error_callback(agent.handle_error)
	coordinator.register_state_change_callback(agent.get_next_action_in_game)
	coordinator.register_out_of_game_callback(agent.get_next_action_out_of_game)

	# We're running an AI, it doesn't make sense to play anything other than defect
	chosenClass = PlayerClass.DEFECT
	agent.change_class(chosenClass)
	print("chosen_class:", getattr(agent, 'chosen_class', None))

	# startコマンド送信のための明示的な呼び出し
	action = agent.get_next_action_out_of_game()
	print("get_next_action_out_of_game() called, action:", action)
	if action:
		action.execute(coordinator)
		runlog_path = r"C:\\Program Files (x86)\\Steam\\steamapps\\common\\SlayTheSpire\\saves\\DEFECT.run.log"
		last_learned_floor = -1

		def runlog_watcher():
			# run.logを1秒ごとに監視し、新しいフロアがあれば学習
			# last_learned_floorをnonlocalで参照
			nonlocal last_learned_floor
			while True:
				time.sleep(1)
				if hasattr(agent, 'learn_new_floor_from_runlog'):
					new_floor = agent.learn_new_floor_from_runlog(runlog_path, last_learned_floor)
					if new_floor > last_learned_floor:
						last_learned_floor = new_floor

		# run.log監視スレッドを起動
		runlog_thread = threading.Thread(target=runlog_watcher, daemon=True)
		runlog_thread.start()

		# メインスレッドでcoordinatorのrunループを開始
		coordinator.run()

def get_class_folder_name(chosenClass: PlayerClass) -> str:
	if chosenClass == PlayerClass.IRONCLAD:
		return "1_IRONCLAD"
	elif chosenClass == PlayerClass.THE_SILENT:
		return "1_THE_SILENT"
	elif chosenClass == PlayerClass.DEFECT:
		return "1_DEFECT"

def copy_run_files(results, chosenClass, folder_name):
	# Quick and dirty grab of run files for a given ascension streak
	# We assume that the CWD is the SlayTheSpire game folder
	repo_base = os.path.join(os.getcwd(), "..")
	game_path = os.path.join(repo_base, "SlayTheSpire")
	mod_path = os.path.join(repo_base, "Mods", "spirecomm")
	# Quick and dirty grab of run files for a given ascension streak
	# We assume that the CWD is the SlayTheSpire game folder
	repo_base = os.path.join(os.getcwd(), "..")
	game_path = os.path.join(repo_base, "SlayTheSpire")
	mod_path = os.path.join(repo_base, "Mods", "spirecomm")

	# We are located in the SlayTheSpire directory by default
	game_runs_path = os.path.join(game_path, "runs", get_class_folder_name(chosenClass))
	mod_runs_path = os.path.join(mod_path, "runs")
	mod_specific_runs_path = os.path.join(mod_runs_path, folder_name)

	logging.info(f"Creating runs folder in mod folder: {mod_runs_path}")
	os.makedirs(mod_runs_path, exist_ok=True)
	logging.info(f"Creating specific run folder in mod runs folder: {mod_specific_runs_path}")
	os.makedirs(mod_specific_runs_path, exist_ok=True)

	if not os.path.exists(game_runs_path):
		logging.error(f"Game runs path does not exist: {game_runs_path}")
		return
	try:
		logging.info("Copying from game runs folder to mod runs folder")
		run_files = os.listdir(game_runs_path)
		run_files.sort()
		run_file_names = run_files[-len(results):]
		for file in run_file_names:
			with open(os.path.join(game_runs_path, file), "r") as source_data:
				with open(os.path.join(mod_specific_runs_path, file), "w") as dest_data:
					dest_data.write(source_data.read())
	except Exception as e:
		logging.error("Ran into a problem while copying run information from game to mod folder: " + str(e))
		logging.info("Copying from game runs folder to mod runs folder")
		run_files = os.listdir(game_runs_path)

		run_files.sort()
		run_file_names = run_files[-len(results):]

		for file in run_file_names:
			with open(os.path.join(game_runs_path, file), "r") as source_data:
				with open(os.path.join(mod_specific_runs_path, file), "w") as dest_data:
					dest_data.write(source_data.read())
	except OSError as e:
		logging.error("Ran into a problem while copying run information from game to mod folder: " + e)

if __name__ == "__main__":
	main()