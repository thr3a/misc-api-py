"""Knights and Knaves パズルの共通ロジック."""

from z3 import Bool, BoolVal, If, Implies, Not, Or, Solver, Sum

STATEMENT_TYPES_EASY = ["accusation", "affirmation", "sympathetic", "antithetic"]
STATEMENT_TYPES_ALL = [
    "accusation", "affirmation", "sympathetic", "antithetic",
    "disjoint", "joint",
    "at_least_knight", "majority_knight", "at_least_knave", "majority_knave",
    "odd_knight", "even_knight", "odd_knave", "even_knave",
    "pair_same", "pair_diff", "if_then", "either_knight", "either_knave",
]
STATEMENT_TYPES_COUNTING = ["at_least_knight", "majority_knight", "at_least_knave", "majority_knave", "odd_knight", "even_knight", "odd_knave", "even_knave"]
STATEMENT_TYPES_TWO_TARGET = ["pair_same", "pair_diff", "if_then", "either_knight", "either_knave"]

_COUNTING_TYPES_SET = set(STATEMENT_TYPES_COUNTING)
_TWO_TARGET_TYPES_SET = set(STATEMENT_TYPES_TWO_TARGET)


def to_text(stmt: dict) -> str:
    """発言をテキストに変換する."""
    sp = stmt["speaker"]
    tg = stmt.get("target")
    t = stmt["type"]

    if t == "accusation":
        return f"{sp}は言った: 「{tg}は悪党だ。」"
    elif t == "affirmation":
        return f"{sp}は言った: 「{tg}は騎士だ。」"
    elif t == "sympathetic":
        return f"{sp}は言った: 「{tg}は私と同じ種類だ。」"
    elif t == "antithetic":
        return f"{sp}は言った: 「{tg}は私と異なる種類だ。」"
    elif t == "disjoint":
        ck = stmt["claimed_is_knight"]
        kind = "騎士" if ck else "悪党"
        return f"{sp}は言った: 「{tg}は{kind}だ、または私は悪党だ。」"
    elif t == "joint":
        ck = stmt["claimed_is_knight"]
        kind = "騎士" if ck else "悪党"
        return f"{sp}は言った: 「{tg}は{kind}だ、そして私は悪党だ。」"
    elif t == "at_least_knight":
        n = stmt["n"]
        return f"{sp}は言った: 「この中に騎士は少なくとも{n}人いる。」"
    elif t == "majority_knight":
        return f"{sp}は言った: 「騎士のほうが多い。」"
    elif t == "at_least_knave":
        n = stmt["n"]
        return f"{sp}は言った: 「この中に悪党は少なくとも{n}人いる。」"
    elif t == "majority_knave":
        return f"{sp}は言った: 「悪党のほうが多い。」"
    elif t == "odd_knight":
        return f"{sp}は言った: 「この中に騎士は奇数人いる。」"
    elif t == "even_knight":
        return f"{sp}は言った: 「この中に騎士は偶数人いる。」"
    elif t == "odd_knave":
        return f"{sp}は言った: 「この中に悪党は奇数人いる。」"
    elif t == "even_knave":
        return f"{sp}は言った: 「この中に悪党は偶数人いる。」"
    elif t == "pair_same":
        tg2 = stmt["target2"]
        return f"{sp}は言った: 「{tg}と{tg2}は同じ種類だ。」"
    elif t == "pair_diff":
        tg2 = stmt["target2"]
        return f"{sp}は言った: 「{tg}と{tg2}は異なる種類だ。」"
    elif t == "if_then":
        tg2 = stmt["target2"]
        ck = stmt["condition_knight"]
        ek = stmt["conclusion_knight"]
        cond_kind = "騎士" if ck else "悪党"
        conc_kind = "騎士" if ek else "悪党"
        return f"{sp}は言った: 「もし{tg}が{cond_kind}なら、{tg2}は{conc_kind}だ。」"
    elif t == "either_knight":
        tg2 = stmt["target2"]
        return f"{sp}は言った: 「{tg}か{tg2}の少なくとも一方は騎士だ。」"
    elif t == "either_knave":
        tg2 = stmt["target2"]
        return f"{sp}は言った: 「{tg}か{tg2}の少なくとも一方は悪党だ。」"
    else:
        raise ValueError(f"未知のタイプ: {t}")


def add_constraint(solver: Solver, stmt: dict, is_knight: dict) -> None:
    """z3ソルバーに発言の制約を追加する.

    各発言タイプの制約:
    - accusation:  speaker と target は異なる種類
    - affirmation: speaker と target は同じ種類
    - sympathetic: target は騎士（speaker の種類に関わらず成立）
    - antithetic:  target は悪党（speaker の種類に関わらず成立）
    - disjoint:    speaker は騎士、target は claimed_is_knight の通り
    - joint:       speaker は悪党、target は claimed_is_knight と逆
    """
    sp = stmt["speaker"]
    tg = stmt.get("target")
    t = stmt["type"]

    if t == "accusation":
        solver.add(is_knight[sp] != is_knight[tg])
    elif t == "affirmation":
        solver.add(is_knight[sp] == is_knight[tg])
    elif t == "sympathetic":
        solver.add(is_knight[tg])
    elif t == "antithetic":
        solver.add(Not(is_knight[tg]))
    elif t == "disjoint":
        ck = stmt["claimed_is_knight"]
        solver.add(is_knight[sp])
        solver.add(is_knight[tg] == BoolVal(ck))
    elif t == "joint":
        ck = stmt["claimed_is_knight"]
        solver.add(Not(is_knight[sp]))
        solver.add(is_knight[tg] == BoolVal(not ck))
    elif t in _COUNTING_TYPES_SET:
        num_persons = len(is_knight)
        knight_count = Sum([If(is_knight[p], 1, 0) for p in is_knight])
        knave_count = num_persons - knight_count
        if t == "at_least_knight":
            n = stmt["n"]
            solver.add(is_knight[sp] == (knight_count >= n))
        elif t == "majority_knight":
            solver.add(is_knight[sp] == (knight_count * 2 > num_persons))
        elif t == "at_least_knave":
            n = stmt["n"]
            solver.add(is_knight[sp] == (knave_count >= n))
        elif t == "majority_knave":
            solver.add(is_knight[sp] == (knave_count * 2 > num_persons))
        elif t == "odd_knight":
            solver.add(is_knight[sp] == (knight_count % 2 == 1))
        elif t == "even_knight":
            solver.add(is_knight[sp] == (knight_count % 2 == 0))
        elif t == "odd_knave":
            solver.add(is_knight[sp] == (knave_count % 2 == 1))
        elif t == "even_knave":
            solver.add(is_knight[sp] == (knave_count % 2 == 0))
    elif t in _TWO_TARGET_TYPES_SET:
        tg2 = stmt["target2"]
        if t == "pair_same":
            solver.add(is_knight[sp] == (is_knight[tg] == is_knight[tg2]))
        elif t == "pair_diff":
            solver.add(is_knight[sp] == (is_knight[tg] != is_knight[tg2]))
        elif t == "if_then":
            ck = stmt["condition_knight"]
            ek = stmt["conclusion_knight"]
            solver.add(is_knight[sp] == Implies(is_knight[tg] == BoolVal(ck), is_knight[tg2] == BoolVal(ek)))
        elif t == "either_knight":
            solver.add(is_knight[sp] == Or(is_knight[tg], is_knight[tg2]))
        elif t == "either_knave":
            solver.add(is_knight[sp] == Or(Not(is_knight[tg]), Not(is_knight[tg2])))
    else:
        raise ValueError(f"未知のタイプ: {t}")


def build_solver(puzzle: dict) -> tuple[Solver, dict]:
    """パズルからz3ソルバーと変数辞書を構築する."""
    persons = puzzle["persons"]
    is_knight = {p: Bool(f"is_knight_{p}") for p in persons}
    solver = Solver()
    for stmt in puzzle["statements"]:
        add_constraint(solver, stmt, is_knight)
    return solver, is_knight
