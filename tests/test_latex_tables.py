"""rec. 14: LaTeX tables are escaped and formatted."""
import pandas as pd

from data_driven_plan import latex_table


def test_headers_and_cells_are_escaped(tmp_path):
    df = pd.DataFrame({"q_i": [0.9], "E[Idle] (min)": [-0.0], "Écart (%)": [12.3456], "A & B": ["x_y"]})
    tex = latex_table(df, str(tmp_path / "t.tex"), caption="$Z^\\star$", label="tab:t", decimals=2)
    assert "q\\_i" in tex and "(\\%)" in tex and "A \\& B" in tex and "x\\_y" in tex
    assert "-0.00" not in tex and "12.35" in tex
    assert "\\toprule" in tex and "\\caption{$Z^\\star$}" in tex
