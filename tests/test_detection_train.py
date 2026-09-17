"""Offline regressions for generated OD notebook, splits and embedded restore."""
import copy
import json
import re
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ops import push_detection_train as push
from src.config import apply_overrides, load_config
from src.data.coco_det import prepare_od_index, validate_od_index


def args(**changes):
    values = dict(config="configs/sandwich_coco_det.yaml", commit="abc123", n_train=4,
                  n_val=2, epochs=13, overrides="", slug="run-a", run_id=None,
                  resume_checkpoint_dir=None)
    values.update(changes)
    return SimpleNamespace(**values)


def source(a):
    return "".join(push.render_notebook(a)["cells"][0]["source"])


def exports(src):
    return {line.split("=", 1)[0].split()[1]: shlex.split(line.split("=", 1)[1])[0]
            for line in src.splitlines() if line.startswith("export OD_")}


def embedded(src):
    return re.findall(r"python - <<'PY'\n(.*?)\nPY", src, re.S)


@pytest.fixture
def coco(tmp_path):
    def annotation(name, ids):
        directory = tmp_path / name
        directory.mkdir()
        path = tmp_path / (name + ".json")
        path.write_text(json.dumps({
            "images": [{"id": i, "file_name": f"{i:012}.jpg", "width": 8, "height": 8} for i in ids],
            "annotations": [{"image_id": i, "bbox": [0, 0, 2, 2], "category_id": 1} for i in ids]}))
        return str(directory), str(path)
    return (*annotation("train2017", range(10)), *annotation("val2017", range(100, 104)),
            str(tmp_path / "index.json"))


def test_notebook_epochs_paths_and_shell_tokens():
    overrides = "loss.beta=0.02 'out_dir=outputs/a b; $HOME' 'data.index=data/a b.json' 'task.teachers=[a, b]'"
    src = source(args(overrides=overrides))
    env = exports(src)
    tokens = json.loads(env["OD_OVERRIDES"])
    train_line = next(line for line in src.splitlines() if line.startswith("python train.py"))
    assert shlex.split(train_line)[4:] == tokens
    cfg = apply_overrides(load_config(env["OD_CONFIG"]), tokens)
    assert cfg["train"]["epochs"] == 13
    assert cfg["task"]["teachers"] == ["a", "b"]
    assert cfg["out_dir"] == "outputs/a b; $HOME"
    out = next(line for line in src.splitlines() if line.startswith("OUT_DIR="))
    assert shlex.split(out.split("=", 1)[1]) == [cfg["out_dir"]]
    assert cfg["data"]["max_train_items"] == 4
    assert cfg["data"]["max_val_items"] == 2
    assert cfg["train"]["resume"] is False
    assert len(embedded(src)) == 2
    for code in embedded(src):
        compile(code, "notebook-cell", "exec")
    assert not re.search(r"__[A-Z_]+__", src), "unsubstituted template placeholder"
    explicit = exports(source(args(overrides="train.epochs=17 loss.beta=0.1")))
    assert apply_overrides({}, json.loads(explicit["OD_OVERRIDES"]))["train"]["epochs"] == 17


def test_generated_index_code_reserves_val2017_and_revalidates_cache(coco, monkeypatch):
    tr, ann_tr, te, ann_te, index_path = coco
    env = dict(TRAIN_DIR=tr, TRAIN_ANN=ann_tr, TEST_DIR=te, TEST_ANN=ann_te,
               INDEX=index_path, N_TRAIN="4", N_VAL="2", SEED="0")
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    code = embedded(source(args()))[0]
    exec(code, {})
    index = json.loads(Path(index_path).read_text())
    assert [len(index[k]) for k in ("train", "val", "test")] == [4, 2, 4]
    assert all(r["image_id"] < 10 for k in ("train", "val") for r in index[k])
    assert {r["image_id"] for r in index["test"]} == set(range(100, 104))
    fingerprint = validate_od_index(index)
    exec(code, {})
    assert validate_od_index(json.loads(Path(index_path).read_text())) == fingerprint
    index["val"][0] = index["train"][0]
    Path(index_path).write_text(json.dumps(index))
    with pytest.raises(ValueError, match="Overlapping"):
        exec(code, {})


@pytest.mark.parametrize("mutation,match", [
    (lambda d: d["meta"].pop("split_policy"), "incompatible"),
    (lambda d: d.update(val=[]), "empty"),
    (lambda d: d["val"].pop(), "expected 2"),
    (lambda d: d["train"].__setitem__(1, d["train"][0]), "Duplicate"),
    (lambda d: d["val"].__setitem__(0, d["test"][0]), "Overlapping"),
    (lambda d: d["val"][0].update(image_id=999), "wrong annotation source"),
])
def test_cached_index_fail_closed(coco, mutation, match):
    prepare_od_index(*coco, n_train=4, n_val=2)
    path = Path(coco[-1])
    index = json.loads(path.read_text())
    mutation(index)
    path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match=match):
        prepare_od_index(*coco, n_train=4, n_val=2)


def test_insufficient_or_invalid_requested_counts(coco):
    for counts in ((9, 2), (0, 2), (4, 0)):
        with pytest.raises(ValueError):
            prepare_od_index(*coco, n_train=counts[0], n_val=counts[1])
        assert not Path(coco[-1]).exists()


@pytest.fixture
def restore_case(coco, tmp_path, monkeypatch):
    prepare_od_index(*coco, n_train=4, n_val=2)
    src_dir, dst_dir = tmp_path / "prior/checkpoints", tmp_path / "new run"
    src_dir.mkdir(parents=True)
    a = args(overrides=shlex.join([f"data.index={coco[-1]}", f"out_dir={dst_dir}"]),
             resume_checkpoint_dir=str(src_dir))
    src = source(a)
    env = exports(src)
    # Resolve relative config locally; notebook runs from repo root.
    env["OD_CONFIG"] = str(push.REPO / env["OD_CONFIG"])
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    cfg = apply_overrides(load_config(env["OD_CONFIG"]), json.loads(env["OD_OVERRIDES"]))
    cfg["data"]["od_split_fingerprint"] = validate_od_index(json.loads(Path(coco[-1]).read_text()))
    best = dict(cfg=copy.deepcopy(cfg), model={"weight": torch.tensor(7)}, epoch=1,
                global_step=5, best_val=0.25, no_improve=0)
    last = dict(cfg=copy.deepcopy(cfg), model={"weight": torch.tensor(9)}, epoch=3,
                global_step=15, best_val=0.25, no_improve=2)
    for st in (best, last):
        st["cfg"]["out_dir"] = "old/location"
        st["cfg"]["train"]["epochs"] = 8
    def save():
        torch.save(best, src_dir / "preprocessor.pth")
        torch.save(last, src_dir / "preprocessor_last.pth")
    save()
    return embedded(src)[1], src_dir, dst_dir / "checkpoints", best, last, save


def test_embedded_restore_preserves_both_files_exactly(restore_case):
    code, src, dst, *_ = restore_case
    exec(code, {})
    for name in ("preprocessor.pth", "preprocessor_last.pth"):
        assert (dst / name).read_bytes() == (src / name).read_bytes()
    assert torch.load(dst / "preprocessor.pth", weights_only=True)["model"]["weight"].item() == 7


@pytest.mark.parametrize("missing", ["preprocessor.pth", "preprocessor_last.pth"])
def test_restore_rejects_missing_pair(restore_case, missing):
    code, src, dst, *_ = restore_case
    (src / missing).unlink()
    with pytest.raises(ValueError, match="historical best"):
        exec(code, {})
    assert not dst.exists()


@pytest.mark.parametrize("change", ["run", "model", "split", "legacy", "best_val"])
def test_restore_rejects_incompatible_pair(restore_case, change):
    code, src, dst, best, last, save = restore_case
    if change == "run":
        last["cfg"]["train"]["od_run_id"] = "unrelated"
    elif change == "model":
        last["cfg"]["model"]["post_base"] = 16
    elif change == "split":
        last["cfg"]["data"]["od_split_fingerprint"] = "other-split"
    elif change == "legacy":
        last["cfg"]["data"].pop("od_split_fingerprint")
    else:
        last["best_val"] = 0.5
    save()
    with pytest.raises(ValueError, match="Incompatible|consistent historical"):
        exec(code, {})
    assert not dst.exists()


def test_restore_does_not_search_ambiguous_dataset_root(restore_case, monkeypatch):
    code, src, dst, *_ = restore_case
    unrelated = src.parent / "unrelated/checkpoints"
    unrelated.mkdir(parents=True)
    for file in src.iterdir():
        (unrelated / file.name).write_bytes(file.read_bytes())
    monkeypatch.setenv("OD_RESUME_DIR", str(src.parent))
    with pytest.raises(ValueError, match="exact checkpoint directory"):
        exec(code, {})
    assert not dst.exists()


def test_fresh_run_ignores_mounted_checkpoints_but_rejects_local_ones(restore_case, monkeypatch):
    code, src, dst, *_ = restore_case
    monkeypatch.setenv("OD_RESUME_DIR", "")
    exec(code, {})
    assert not dst.exists()
    dst.mkdir(parents=True)
    (dst / "preprocessor.pth").write_bytes(b"existing")
    with pytest.raises(ValueError, match="Existing local checkpoints"):
        exec(code, {})


def test_engine_rejects_invalid_index_before_building_models(coco, monkeypatch):
    from src import engine

    prepare_od_index(*coco, n_train=4, n_val=2)
    path = Path(coco[-1])
    index = json.loads(path.read_text())
    index["val"] = []
    path.write_text(json.dumps(index))
    monkeypatch.setattr(engine, "_build_models", lambda *a: pytest.fail("models built before validation"))
    with pytest.raises(ValueError, match="empty"):
        engine._train_detection({"data": {"index": str(path)}, "train": {}})


def test_engine_rejects_zero_training_batches(coco, monkeypatch):
    from src import engine

    prepare_od_index(*coco, n_train=4, n_val=2)
    monkeypatch.setattr(engine, "_build_models", lambda *a: (None, None, None))
    with pytest.raises(ValueError, match="at least one full training batch"):
        engine._train_detection({"data": {"index": coco[-1]}, "train": {"batch_size": 8}})


def test_restore_refuses_to_overwrite_existing_destination(restore_case):
    code, src, dst, *_ = restore_case
    dst.mkdir(parents=True)
    file = dst / "preprocessor.pth"
    file.write_bytes(b"unrelated best")
    with pytest.raises(ValueError, match="Destination already has checkpoints"):
        exec(code, {})
    assert file.read_bytes() == b"unrelated best"
    assert not (dst / "preprocessor_last.pth").exists()


@pytest.mark.parametrize("overrides", ["train.resume=true", "data.max_val_items=99", "train.finetune=true"])
def test_pusher_rejects_conflicting_controls(overrides):
    with pytest.raises(ValueError):
        source(args(overrides=overrides))
