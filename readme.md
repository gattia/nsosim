```bash
# create environment
conda create -n nsosim python=3.9
conda activate nsosim

# install NSM 
mkdir dependencies
cd dependencies
git clone https://github.com/gattia/nsm.git
cd nsm
mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
make requirements
pip install .

cd ../..
pip install -r requirements.txt

pip install -e .

```