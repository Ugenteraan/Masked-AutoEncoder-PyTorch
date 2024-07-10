from PIL import Image
import os

def check_images(root_dir):
    for root, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Bad file: {file_path} - {e}")
                # os.remove(file_path)

check_images('./cat/')
