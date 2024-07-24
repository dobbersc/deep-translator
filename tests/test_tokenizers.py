from typing import Literal

import pytest

from translator.tokenizers import preprocess


@pytest.mark.parametrize(
    ("segmentation_level", "expected"),
    [
        ("word", "ant\u00F3nio manuel de oliveira guterres is a portuguese politician and diplomat .".split()),
        ("character", list("ant\u00F3nio manuel de oliveira guterres is a portuguese politician and diplomat.")),
    ],
)
def test_preprocess(segmentation_level: Literal["word", "subword", "character"], expected: list[str]) -> None:
    result: list[str] = preprocess(
        "Anto\u0301nio Manuel de Oliveira Guterres is a Portuguese politician and diplomat.",
        language="en",
        unicode_normalization="NFKC",
        segmentation_level=segmentation_level,
        lowercase=True,
    )
    assert result == expected
