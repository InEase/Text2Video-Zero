export http_proxy=http://192.168.1.174:12798 && export https_proxy=http://192.168.1.174:12798

conda create  -y -n iCen python=3.9
conda activate iCen

git clone https://github.com/InEase/Text2Video-Zero.git
cd Text2Video-Zero/

pip install -r requirements.txt

unset http_proxy
unset https_proxy

python app.py