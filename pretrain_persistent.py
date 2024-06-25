'''A workaround to segmentation fault (core dumped) error.
'''



import subprocess 
import time 


def run_script():

	while True:

		process = subprocess.Popen(['python', 'pretrain.py', '--logging_config', 'configs/pretrain/logging_pretrain.yaml', '--config', 'configs/pretrain/mae_pretrain_224_16.yaml'])

		process.wait()

		if process.returncode == 139:
			print("Segmentation fault occured, restarting script...")
			time.sleep(2)

		else:
			print("Process has been stopped, exiting...")
			break


if __name__ == '__main__':
	run_script()