# ABOUTME: Offline tests for organism discovery: experiment grouping from dated train-config
# ABOUTME: names, building an Organism from its files, the menu, picks and dated arm naming.

import pytest

from src.chat.organisms import (
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
        (
            "2026-08-06_lora_qwen36_table2_9284_difficult_advice_716_dynbatch",
            ("table2_9284", "difficult_advice_716"),
        ),
        (
            "2026-08-17_lora_qwen36_table2_9284_synthdoc_716_less_swap_dynbatch",
            ("table2_9284", "synthdoc_716_less_swap"),
        ),
        (
            "2026-08-25_lora_qwen36_table2_9284_gpt_responder_685_paired",
            ("table2_9284", "gpt_responder_685_paired"),
        ),
        ("2026-08-25_lora_qwen36_table2_synthdoc", ("table2", "synthdoc")),
        ("2026-08-25_lora_qwen36_table2_80_memself_20", ("table2", "80_memself_20")),
        ("2026-08-19_lora_qwen36_less_top_10_220_rank_64", ("less", "top_10_220")),
        ("2026-08-02_lora_qwen36_synthdoc_20_80.yaml", ("synthdoc", "20_80")),
        (
            "configs/train/2026-07-31_lora_qwen3_difficult_advice_thinking.yaml",
            ("difficult", "advice_thinking"),
        ),
        ("odd", ("odd", "odd")),
    ],
)
def test_experiment_group_strips_model_prefix_and_hardware_noise(config, expected):
    assert experiment_group(config) == expected


def test_organism_from_files_reads_base_from_adapter_config_and_mode_from_stamp():
    o = organism_from_files(
        "LASR-Callum/2026-08-14-difficult-advice-716-fixture",
        ADAPTER,
        stamp("2026-08-06_lora_qwen36_table2_9284_difficult_advice_716"),
        "2026-08-15T00:00:00",
    )
    assert (o.base_model, o.mode, o.lora_rank) == ("Qwen/Qwen3.6-27B", "think", 64)
    assert (o.group, o.variant, o.trained) == (
        "table2_9284", "difficult_advice_716", "2026-08-14")
    # An organism trained under an undated adapter repo is still named with its date.
    assert o.name == "2026-08-14_difficult_advice_716_fixture"
    assert o.key == o.name and o.unservable == ""
    assert o.label == "difficult advice 716 fixture (2026-08-14)"


def test_organism_with_a_local_base_path_falls_back_to_the_stamp_then_is_unservable():
    local = {"base_model_name_or_path": "/root/qwen36", "r": 64}
    ok = organism_from_files("o/a", local, stamp("2026-08-02_lora_qwen36_synthdoc_20_80"), "")
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
            org("o/da716", "2026-08-06_lora_qwen36_table2_9284_difficult_advice_716"),
            org("o/synthdoc", "2026-08-25_lora_qwen36_table2_synthdoc"),
            org("o/broken", "2026-08-06_lora_qwen36_table2_9284_broken",
                unservable="no stamp"),
            org("o/nothink", "2026-08-06_lora_qwen36_table2_9284_difficult_advice_716",
                mode="nothink"),
        ]
    )
    text, numbered = render_menu(organisms, unstamped=3)
    assert [o.repo for o in numbered] == ["o/nothink", "o/synthdoc", "o/da716"]
    assert text.index("Qwen/Qwen3.6-27B · nothink") < text.index(
        "Qwen/Qwen3.6-27B · think"
    )
    assert "[ ×]" in text and "no stamp" in text
    assert "  table2_9284" in text and "  table2" in text
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
        [org("o/a", "2026-08-06_lora_qwen36_table2_9284_a"),
         org("o/b", "2026-08-06_lora_qwen36_table2_9284_b")]
    )
    with pytest.raises(ValueError, match="one base model in one thinking mode"):
        check_one_server([org("o/a", "c"), org("o/b", "c", mode="nothink")])


def test_arm_names_carry_the_date_and_are_disambiguated_by_experiment():
    picked = [
        org("o/a", "2026-08-06_lora_qwen36_table2_9284_synthdoc_716"),
        org("o/b", "2026-08-25_lora_qwen36_table2_synthdoc_716"),
        org("o/c", "2026-08-06_lora_qwen36_table2_9284_difficult_advice_716"),
    ]
    picked[1] = Organism(**{**picked[1].__dict__, "trained": "2026-08-25"})
    assert arm_names(picked) == {
        "o/a": "2026-08-14_synthdoc_716",
        "o/b": "2026-08-25_synthdoc_716",
        "o/c": "2026-08-14_difficult_advice_716",
    }


def test_arm_names_refuse_a_set_that_is_still_ambiguous():
    same_day = [
        org("o/a", "2026-08-06_lora_qwen36_table2_9284_synthdoc_716"),
        org("o/b", "2026-08-06_lora_qwen36_table2_9284_synthdoc_716"),
    ]
    with pytest.raises(ValueError, match="not distinct names"):
        arm_names(same_day)
