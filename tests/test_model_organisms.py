# ABOUTME: Offline tests for organism discovery: experiment grouping from train-config names,
# ABOUTME: building an Organism from its two metadata files, the menu, picks and arm naming.

import pytest

from src.endpoints.model_organisms import (
    Organism,
    arm_names,
    check_one_server,
    experiment_group,
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
    repo: str, config: str, base="Qwen/Qwen3.6-27B", mode="think", unservable=""
) -> Organism:
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
    "config, expected",
    [
        ("lora_qwen36_t2_9284_da716_dynbatch_2xh200", ("t2_9284", "da716")),
        (
            "lora_qwen36_t2_9284_synthdoc716_lessswap_dynbatch_2xh200",
            ("t2_9284", "synthdoc716_lessswap"),
        ),
        (
            "lora_qwen36_t2_9284_grokresp703_paired_2xh200",
            ("t2_9284", "grokresp703_paired"),
        ),
        ("lora_qwen36_table2_synthdoc_h200x4", ("table2", "synthdoc")),
        ("lora_qwen36_table2_80_memself_20", ("table2", "80_memself_20")),
        ("lora_qwen36_less_top10_220_r64", ("less", "top10_220")),
        ("lora_qwen36_less_random220_control_r64", ("less", "random220_control")),
        ("lora_qwen36_synthdoc_20_80.yaml", ("synthdoc", "20_80")),
        (
            "configs/train/lora_qwen3_difficult_advice_thinking.yaml",
            ("difficult", "advice_thinking"),
        ),
        ("odd", ("odd", "odd")),
    ],
)
def test_experiment_group_strips_model_prefix_and_hardware_noise(config, expected):
    assert experiment_group(config) == expected


def test_organism_from_files_reads_base_from_adapter_config_and_mode_from_stamp():
    o = organism_from_files(
        "LASR-Callum/x-lora",
        ADAPTER,
        stamp("lora_qwen36_t2_9284_da716"),
        "2026-08-15T00:00:00",
    )
    assert (o.base_model, o.mode, o.lora_rank) == ("Qwen/Qwen3.6-27B", "think", 64)
    assert (o.group, o.variant, o.trained) == ("t2_9284", "da716", "2026-08-14")
    assert o.key == "x-lora" and o.unservable == ""


def test_organism_with_a_local_base_path_falls_back_to_the_stamp_then_is_unservable():
    local = {"base_model_name_or_path": "/root/qwen36", "r": 64}
    ok = organism_from_files("o/a", local, stamp("lora_qwen36_synthdoc_20_80"), "")
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


def test_render_menu_groups_by_base_mode_and_experiment_and_numbers_only_servable():
    organisms = sort_for_menu(
        [
            org("o/da716", "lora_qwen36_t2_9284_da716_dynbatch_2xh200"),
            org("o/synthdoc", "lora_qwen36_table2_synthdoc_h200x4"),
            org("o/broken", "lora_qwen36_t2_9284_broken", unservable="no stamp"),
            org("o/nothink", "lora_qwen36_t2_9284_da716", mode="nothink"),
        ]
    )
    text, numbered = render_menu(organisms, unstamped=3)
    assert [o.repo for o in numbered] == ["o/nothink", "o/da716", "o/synthdoc"]
    assert text.index("Qwen/Qwen3.6-27B · nothink") < text.index(
        "Qwen/Qwen3.6-27B · think"
    )
    assert "[ ×]" in text and "no stamp" in text
    assert "  t2_9284" in text and "  table2" in text
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
    check_one_server(
        [org("o/a", "lora_qwen36_t2_9284_a"), org("o/b", "lora_qwen36_t2_9284_b")]
    )
    with pytest.raises(ValueError, match="one base model in one thinking mode"):
        check_one_server([org("o/a", "c"), org("o/b", "c", mode="nothink")])


def test_arm_names_are_variants_disambiguated_by_experiment_and_never_base():
    picked = [
        org("o/a", "lora_qwen36_t2_9284_synthdoc_716"),
        org("o/b", "lora_qwen36_table2_synthdoc_716"),
        org("o/c", "lora_qwen36_t2_9284_da716"),
        org("o/d", "lora_qwen36_x_base"),
    ]
    assert arm_names(picked) == {
        "o/a": "t2_9284_synthdoc_716",
        "o/b": "table2_synthdoc_716",
        "o/c": "da716",
        "o/d": "x_base",
    }
