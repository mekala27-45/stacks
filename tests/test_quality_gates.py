import pytest

from scripts.quality_gates import ROOT, palette_checks, punctuation_errors, vocabulary_errors


@pytest.mark.parametrize("check", [punctuation_errors, vocabulary_errors])
def test_authored_gates_clean_and_empty_are_distinct(tmp_path, check):
    with pytest.raises(ValueError, match="No nonempty"):
        check(tmp_path)
    (tmp_path / "note.md").write_text("A measured result with a stated limit.", encoding="utf-8")
    assert check(tmp_path) == []


def test_deliberate_punctuation_violation(tmp_path):
    (tmp_path / "note.md").write_text("Changed " + chr(0x2014) + " deliberately", encoding="utf-8")
    assert punctuation_errors(tmp_path) == ["note.md:1"]


def test_deliberate_vocabulary_violation(tmp_path):
    (tmp_path / "note.md").write_text("A seam" + "less experience", encoding="utf-8")
    assert vocabulary_errors(tmp_path) == ["note.md:1"]


def test_palette_checks_actual_both_theme_tokens():
    checks = palette_checks((ROOT / "web/app/globals.css").read_text(encoding="utf-8"))
    assert {check["theme"] for check in checks} == {"light", "dark"}
    assert checks and all(check["passed"] for check in checks)


def test_palette_rejects_deliberately_invisible_text():
    css = (ROOT / "web/app/globals.css").read_text(encoding="utf-8")
    css = css.replace("--ink: #1b1714;", "--ink: #faf7f1;")
    assert any(not check["passed"] for check in palette_checks(css))


def test_palette_refuses_empty_input():
    with pytest.raises(ValueError, match="Missing"):
        palette_checks("")
