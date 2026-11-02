"""CPU checks for the answer grader and the two update rules."""

from mmk12_zo.scoring import grade
from mmk12_zo.search import es_weights, select_directions


def main() -> None:
    exact = grade(r"work \boxed{20}", "20")
    assert exact.reward == 1 and exact.format_ok == 1
    latex = grade(r"\boxed{4}", "$$4$$")
    assert latex.reward == 1
    unit = grade(r"\boxed{8}", "8cm")
    assert unit.reward == 1
    missing = grade("the answer is 20", "20")
    assert missing.reward == 0 and missing.format_ok == 0
    wrong = grade(r"\boxed{3}", "20")
    assert wrong.reward == 0 and wrong.format_ok == 1
    picked = select_directions([0.1, 0.4, 0.4], [0.0, 0.1, 0.3], top_k=2)
    assert picked["selected"] == [1, 2]
    assert picked["skip"] is False
    flat = es_weights([0.2, 0.2, 0.2])
    assert flat["skip"] and flat["skip_reason"] == "zero_variance"
    varied = es_weights([0.0, 0.5, 1.0])
    assert varied["skip"] is False
    assert abs(sum(varied["weights"])) < 1e-9
    print("cpu checks passed")


if __name__ == "__main__":
    main()
