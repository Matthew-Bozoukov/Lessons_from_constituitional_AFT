# ABOUTME: Offline tests for organism discovery: reading model, style and seed out of an
# ABOUTME: organism's own name, building one from its files, the menu, picks and arm naming.

import pytest

from src.chat.organisms import (
    Organism,
    arm_names,
    check_one_server,
    organism_parts,
    organism_from_files,
    parse_pick,
    render_menu,
    sort_for_menu,
)

ADAPTER = {"base_model_name_or_path": "Qwen/Qwen3.6-27B", "r": 64}


def stamp(config: str, thinking=True, **extra) -> dict:
    return {
        "thinking": thinking,
        "train_config": f"configs/train/{config}.yaml",
        "base_model": "Qwen/Qwen3.6-27B",
        "timestamp": "20260814_101010",
        **extra,
    }


def org(
    repo: str, config: str = "qwen36_difficult_advice", base="Qwen/Qwen3.6-27B",
    mode="think", unservable=""
) -> Organism:
    """An organism whose REPO is its name — which is where group and variant come from."""
    return Organism(
        repo=repo,
        base_model=base,
        mode=mode,
        train_config=config,
        dataset="",
        trained="2026-08-14",
        lora_rank=64,
        unservable=unservable,
    )


@pytest.mark.parametrize(
    "name, expected",
    [
        # The law puts model, style and seed in fixed positions, so nothing here guesses
        # which token is which — it reads them off.
        ("2026-09-04_qwen36_difficult_advice_0", ("qwen36", "difficult_advice", "0")),
        ("2026-09-04_qwen36_difficult_advice_716_coherence_2",
         ("qwen36", "difficult_advice_716_coherence", "2")),
        ("2026-09-04_qwen3_tulu_100_0", ("qwen3", "tulu_100", "0")),
        # From before the law: no registered model to lead with and no seed to find, so
        # it is all style. It still groups and lists; it just groups alone.
        ("2026-08-16_qwen3627b_lora_t2_9284_synthdoc_716_r64",
         ("", "qwen3627b_lora_t2_9284_synthdoc_716_r64",
          "qwen3627b_lora_t2_9284_synthdoc_716_r64")),
    ],
)
def test_an_organisms_parts_are_read_off_its_own_name(name, expected):
    assert organism_parts(name) == expected


def test_organism_from_files_reads_base_from_adapter_config_and_mode_from_stamp():
    o = organism_from_files(
        "LASR-Callum/2026-08-14-qwen36-difficult-advice-716-0",
        ADAPTER,
        stamp("qwen36_difficult_advice_716"),
        "2026-08-15T00:00:00",
    )
    assert (o.base_model, o.mode, o.lora_rank) == ("Qwen/Qwen3.6-27B", "think", 64)
    # group and variant come from the organism's OWN name, not from the config path the
    # stamp records: configs are undated and edited in place, so the path is provenance,
    # never identity.
    assert (o.group, o.variant, o.trained) == (
        "difficult_advice_716", "0", "2026-08-14")
    # The name is the one the training run minted — read, not rebuilt.
    assert o.name == "2026-08-14_qwen36_difficult_advice_716_0"
    assert o.key == o.name and o.unservable == ""
    assert o.label == "qwen36 difficult advice 716 0 (2026-08-14)"


def test_organism_with_a_local_base_path_falls_back_to_the_stamp_then_is_unservable():
    local = {"base_model_name_or_path": "/root/qwen36", "r": 64}
    ok = organism_from_files("o/a", local, stamp("qwen36_synthdoc_20_80"), "")
    assert ok.base_model == "Qwen/Qwen3.6-27B" and not ok.unservable
    bad = organism_from_files(
        "o/b", local, {**stamp("c"), "base_model": "/root/qwen36"}, ""
    )
    assert "local path" in bad.unservable


def test_organism_without_a_boolean_thinking_stamp_is_unservable():
    o = organism_from_files(
        "o/c", ADAPTER, {"train_config": "x", "thinking": "yes"}, "2026-08-01"
    )
    assert "thinking" in o.unservable and o.trained == "2026-08-01"


def test_render_menu_groups_by_base_mode_and_style_and_numbers_only_servable():
    organisms = sort_for_menu(
        [
            org("o/2026-09-04-qwen36-difficult-advice-0"),
            org("o/2026-09-04-qwen36-synthdoc-716-0"),
            org("o/2026-09-04-qwen36-broken-0", unservable="no stamp"),
            org("o/2026-09-04-qwen36-difficult-advice-1", mode="nothink"),
        ]
    )
    text, numbered = render_menu(organisms, unstamped=3)
    assert [o.repo for o in numbered] == [
        "o/2026-09-04-qwen36-difficult-advice-1",
        "o/2026-09-04-qwen36-difficult-advice-0",
        "o/2026-09-04-qwen36-synthdoc-716-0",
    ]
    assert text.index("Qwen/Qwen3.6-27B · nothink") < text.index(
        "Qwen/Qwen3.6-27B · think"
    )
    assert "[ ×]" in text and "no stamp" in text
    # grouped by style-type, with the seeds listed under it
    assert "  difficult_advice" in text and "  synthdoc_716" in text
    assert "+3 adapters without training_meta.json" in text


def test_parse_pick_accepts_lists_ranges_and_quit():
    assert parse_pick("1 3", 5) == [0, 2]
    assert parse_pick("2-4,1", 5) == [1, 2, 3, 0]
    assert parse_pick("q", 5) == [] and parse_pick("", 5) == []
    with pytest.raises(ValueError, match="not on the menu"):
        parse_pick("6", 5)
    with pytest.raises(ValueError, match="not a number"):
        parse_pick("two", 5)


def test_check_one_server_refuses_mixed_headings():
    check_one_server([org("o/2026-09-04-qwen36-difficult-advice-0"),
                      org("o/2026-09-04-qwen36-difficult-advice-1")])
    with pytest.raises(ValueError, match="one base model in one thinking mode"):
        check_one_server([org("o/2026-09-04-qwen36-difficult-advice-0"),
                          org("o/2026-09-04-qwen36-difficult-advice-1", mode="nothink")])


def test_arm_names_are_the_short_handle_anyone_would_say_out_loud():
    picked = [
        org("o/2026-09-04-qwen36-difficult-advice-0"),
        org("o/2026-09-04-qwen36-difficult-advice-1"),
        org("o/2026-09-04-qwen36-synthdoc-716-0"),
    ]
    assert arm_names(picked) == {
        "o/2026-09-04-qwen36-difficult-advice-0": "difficult_advice_0",
        "o/2026-09-04-qwen36-difficult-advice-1": "difficult_advice_1",
        "o/2026-09-04-qwen36-synthdoc-716-0": "synthdoc_716_0",
    }


def test_the_whole_set_falls_back_to_full_names_when_the_short_handle_collides():
    """One arm retrained on another day: the short handle is exactly where it is unsafe."""
    picked = [
        org("o/2026-09-04-qwen36-difficult-advice-0"),
        org("o/2026-09-06-qwen36-difficult-advice-0"),
    ]
    assert arm_names(picked) == {
        "o/2026-09-04-qwen36-difficult-advice-0": "2026-09-04_qwen36_difficult_advice_0",
        "o/2026-09-06-qwen36-difficult-advice-0": "2026-09-06_qwen36_difficult_advice_0",
    }
