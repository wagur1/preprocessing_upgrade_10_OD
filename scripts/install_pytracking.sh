#!/usr/bin/env bash
# Optional: install pytracking to run the paper's exact GOT-10k trackers
# (KYS / DiMP / ATOM / PrDiMP). The default self-contained SiamFC tracker needs
# NONE of this -- use it only if you want the paper's exact tracker numbers.
#
# pytracking is sensitive to torch/CUDA versions; on Kaggle it may need extra
# fiddling. Run at your own pace and check the upstream install guide:
#   https://github.com/visionml/pytracking/blob/master/INSTALL.md
set -e

echo "[install_pytracking] system deps (ninja for PreciseRoIPooling)"
apt-get update -y && apt-get install -y ninja-build libturbojpeg0-dev || true

echo "[install_pytracking] python deps"
pip install -q jpeg4py visdom tb-nightly scikit-image tikzplotlib pandas || true

echo "[install_pytracking] clone repo"
if [ ! -d pytracking ]; then
  git clone https://github.com/visionml/pytracking.git
  cd pytracking
  git submodule update --init --recursive
  cd ..
fi

echo "[install_pytracking] make it importable"
export PYTHONPATH="$PWD/pytracking:$PYTHONPATH"

cat <<'NOTE'
[install_pytracking] Next steps (manual, per upstream):
  1) create pytracking/pytracking/evaluation/local.py and ltr/admin/local.py
     (copy the *_example.py templates), setting dataset + network paths.
  2) download the tracker network weights into pytracking/pytracking/networks/
     e.g. dimp50.pth, atom_default.pth, kys.pth, prdimp50.pth
  3) verify:  python -c "from pytracking.evaluation import Tracker; print('ok')"
Then in eval, select the tracker via config: task.tracker=pytracking:dimp:dimp50
NOTE
