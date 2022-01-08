## Requires Deep Learning AMI (Ubuntu 18.04) Version 53.0 or better. Free tier does not support deeplearning models.
## Need to be redone using tensorflow lite and BERT mobile for ec2 free deployment.
## Alternatively, can use a serverless function (amazon lambda) to deploy the model and interact with EC2 free.
## However, this approach will be slower as serverless lambda functions run on demand.

sudo apt-get update

## Install Python via Anaconda3
###############################
sudo apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
wget https://repo.anaconda.com/archive/Anaconda3-5.0.1-Linux-x86_64.sh
sudo reboot

### Minimum Install Requirements (no Jupyter Notebook)
sudo apt-get install python3-bs4
conda install selenium
sudo apt-get install chromium-chromedriver

pip install requests
pip install torch
pip install tensorflow
pip install flask
pip install sentence-transformers
pip install scipy
pip install keras
pip install --upgrade pandas

#### Clone and generate embeddings if no embeddings exist
git clone https://github.com/lifebringer/CourseProject
cd CourseProject/otaku_search_engine/
FILE=/embeddings/otaku_embeddings.pkl
if [ -f "$FILE" ]; then
    echo "$FILE exists."
else 
    echo "$FILE does not exist."
    python build_search_index.py
fi
#python build_search_index.py

### Start web server
nohup python otaku_semantic_search.py &