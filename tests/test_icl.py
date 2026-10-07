from tdnv.icl import Item, render, sample

TRUTH = [Item(f"s{i}", *(("True", "False") if i % 2 else ("False", "True"))) for i in range(40)]
CAA = [Item(f"q{i}", "A" if i % 2 else "B", "B" if i % 2 else "A") for i in range(40)]


def test_truth_sampling_is_balanced_and_excludes_query():
    insts = sample("truth_x", TRUTH, n_query=10, k_max=15, seed=0)
    assert sum(i.query.y1 == "True" for i in insts) == 5
    for inst in insts:
        assert inst.query not in inst.demos and len(inst.demos) == 15
        n_true = sum(d.y1 == "True" for d in inst.demos)
        assert n_true in (7, 8)


def test_tasks_share_inputs_and_flip_labels():
    inst = sample("truth_x", TRUTH, n_query=2, k_max=5, seed=1)[0]
    p1, p0 = (render("truth_x", inst, 5, t, "<s>") for t in (1, 0))
    assert p1.replace("True", "#").replace("False", "True").replace("#", "False") == p0
    assert p1.endswith(f"Statement: {inst.query.x}\nAnswer:")
    assert render("truth_x", inst, 0, 1, "<s>") == render("truth_x", inst, 0, 0, "<s>")


def test_k_shot_prompts_are_nested():
    inst = sample("caa_x", CAA, n_query=3, k_max=15, seed=2)[0]
    q = render("caa_x", inst, 0, 1, "")
    shots = [render("caa_x", inst, k, 1, "").removesuffix(q) for k in (1, 5, 10, 15)]
    assert all(b.startswith(a) for a, b in zip(shots, shots[1:]))
    assert shots[0] == f"{inst.demos[0].x}\nAnswer: ({inst.demos[0].y1})\n\n"
    assert q.endswith("\nAnswer: (")
