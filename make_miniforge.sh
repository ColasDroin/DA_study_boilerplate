wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b  -p ./miniforge -f
rm -f Miniforge3-Linux-x86_64.sh
source miniforge/bin/activate
python -m pip install ipython numpy scipy pandas fastparquet psutil cpymad xsuite
mkdir modules_alt
cd modules_alt
git clone https://github.com/xsuite/tree_maker.git
python -m pip install -e tree_maker
git clone https://github.com/xsuite/xmask.git
pip install -e xmask
cd xmask/
git submodule init
git submodule update
cd ../../
xsuite-prebuild


