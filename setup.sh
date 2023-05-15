conda init

# ---------------

export http_proxy=http://192.168.1.174:12798 && export https_proxy=http://192.168.1.174:12798

conda create  -y -n iCen python=3.9
conda activate iCen

git clone https://github.com/InEase/Text2Video-Zero.git --depth 1
cd Text2Video-Zero/

unset http_proxy
unset https_proxy

pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

python app.py
