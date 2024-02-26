# Define the method for get models
import os

SERVER_URL = "192.168.1.60"
FTP_USER = "anonymous"
FTP_PASS = "anonymous"

def dload_model(model_name, dst_path="model_data"):
    if os.path.exists(f"{dst_path}/{model_name}"):
        return
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    # download the model from the server by ftp with anonymous login
    os.system(f"wget -nv --recursive --no-parent -nH --cut-dirs=2 ftp://{FTP_USER}:{FTP_PASS}@{SERVER_URL}/pub/models/{model_name} -P {dst_path}")

# script selftest
if __name__ == "__main__":
    dload_model("mnist", "model")