"""Knights and Knaves パズルの生成・解答サービス."""

import random

from z3 import And, Bool, Not, sat
from z3 import is_true as z3_is_true

from .puzzle_lib import (
    _COUNTING_TYPES_SET,
    _TWO_TARGET_TYPES_SET,
    STATEMENT_TYPES_ALL,
    STATEMENT_TYPES_EASY,
    add_constraint,
    build_solver,
    to_text,
)

MAX_VISIBLE_COUNT_N = 4


def solve_puzzle(puzzle: dict) -> dict[str, str] | None:
    """z3でパズルを解いて解答辞書を返す.

    Args:
        puzzle: persons と statements を含むパズル辞書

    Returns:
        {人名: "knight" | "knave"} の辞書、解なしの場合は None
    """
    solver, is_knight = build_solver(puzzle)

    if solver.check() != sat:
        return None

    model = solver.model()
    return {p: ("knight" if z3_is_true(model.eval(is_knight[p])) else "knave") for p in puzzle["persons"]}


def to_text_list(statements: list[dict]) -> list[str]:
    """発言リストをテキストリストに変換する.

    Args:
        statements: 発言辞書のリスト

    Returns:
        テキスト化された発言のリスト
    """
    return [to_text(s) for s in statements]


def _is_type_valid(t: str, sp_is_knight: bool, tg_is_knight: bool) -> bool:
    """発言タイプが解と整合するか確認する."""
    if t == "accusation":
        return sp_is_knight != tg_is_knight
    elif t == "affirmation":
        return sp_is_knight == tg_is_knight
    elif t == "sympathetic":
        return tg_is_knight
    elif t == "antithetic":
        return not tg_is_knight
    elif t == "disjoint":
        return sp_is_knight  # 騎士のみ
    elif t == "joint":
        return not sp_is_knight  # 悪党のみ
    return False


def _make_counting_statement(
    t: str,
    speaker: str,
    sp_is_knight: bool,
    knight_count: int,
    num_persons: int,
) -> dict | None:
    """カウント系発言dictを生成する. 生成不可の場合は None を返す."""
    knave_count = num_persons - knight_count
    visible_cap = min(MAX_VISIBLE_COUNT_N, num_persons)
    stmt: dict = {"type": t, "speaker": speaker}

    if t == "at_least_knight":
        if sp_is_knight:
            if knight_count == 0:
                return None  # 「少なくとも0人」は自明で使わない
            candidates = list(range(1, min(knight_count, visible_cap) + 1))
        else:
            candidates = list(range(knight_count + 1, visible_cap + 1))
        if not candidates:
            return None
        stmt["n"] = random.choice(candidates)
    elif t == "majority_knight":
        claim_true = knight_count * 2 > num_persons
        if sp_is_knight != claim_true:
            return None
    elif t == "at_least_knave":
        if sp_is_knight:
            if knave_count == 0:
                return None
            candidates = list(range(1, min(knave_count, visible_cap) + 1))
        else:
            candidates = list(range(knave_count + 1, visible_cap + 1))
        if not candidates:
            return None
        stmt["n"] = random.choice(candidates)
    elif t == "majority_knave":
        claim_true = knave_count * 2 > num_persons
        if sp_is_knight != claim_true:
            return None
    elif t == "odd_knight":
        claim_true = knight_count % 2 == 1
        if sp_is_knight != claim_true:
            return None
    elif t == "even_knight":
        claim_true = knight_count % 2 == 0
        if sp_is_knight != claim_true:
            return None
    elif t == "odd_knave":
        claim_true = knave_count % 2 == 1
        if sp_is_knight != claim_true:
            return None
    elif t == "even_knave":
        claim_true = knave_count % 2 == 0
        if sp_is_knight != claim_true:
            return None

    return stmt


def _make_two_target_statement(
    t: str,
    speaker: str,
    target1: str,
    target2: str,
    sp_is_knight: bool,
    tg1_is_knight: bool,
    tg2_is_knight: bool,
) -> dict | None:
    """2ターゲット型発言dictを生成する. 生成不可の場合は None を返す."""
    stmt: dict = {"type": t, "speaker": speaker, "target": target1, "target2": target2}

    if t == "pair_same":
        claim_true = tg1_is_knight == tg2_is_knight
        if sp_is_knight != claim_true:
            return None
    elif t == "pair_diff":
        claim_true = tg1_is_knight != tg2_is_knight
        if sp_is_knight != claim_true:
            return None
    elif t == "if_then":
        combos = [(ck, ek) for ck in [True, False] for ek in [True, False]]
        random.shuffle(combos)
        for ck, ek in combos:
            antecedent = tg1_is_knight == ck
            consequent = tg2_is_knight == ek
            claim_true = (not antecedent) or consequent
            if sp_is_knight == claim_true:
                stmt["condition_knight"] = ck
                stmt["conclusion_knight"] = ek
                return stmt
        return None
    elif t == "either_knight":
        claim_true = tg1_is_knight or tg2_is_knight
        if sp_is_knight != claim_true:
            return None
    elif t == "either_knave":
        claim_true = (not tg1_is_knight) or (not tg2_is_knight)
        if sp_is_knight != claim_true:
            return None

    return stmt


def _make_statement(t: str, speaker: str, target: str, tg_is_knight: bool) -> dict:
    """発言dictを生成する."""
    stmt: dict = {"type": t, "speaker": speaker, "target": target}
    if t == "disjoint":
        stmt["claimed_is_knight"] = tg_is_knight  # 真実を述べる
    elif t == "joint":
        stmt["claimed_is_knight"] = not tg_is_knight  # 嘘をつく
    return stmt


def _generate_statements(
    persons: list[str],
    solution: dict[str, str],
    types: list[str],
    num_statements: int,
) -> list[dict]:
    """解と整合する発言リストを生成する."""
    speakers = list(persons)
    extra = num_statements - len(speakers)
    for _ in range(extra):
        speakers.append(random.choice(persons))
    random.shuffle(speakers)

    knight_count = sum(1 for p in persons if solution[p] == "knight")
    num_persons = len(persons)

    majority_pair_used = False
    at_least_knight_used = False
    at_least_knave_used = False
    parity_group_used = False
    if_then_used = False

    statements = []
    for speaker in speakers:
        targets = [p for p in persons if p != speaker]
        target = random.choice(targets)

        sp_is_knight = solution[speaker] == "knight"
        tg_is_knight = solution[target] == "knight"

        if len(targets) >= 2:
            remaining = [p for p in targets if p != target]
            target2 = random.choice(remaining)
            tg2_is_knight = solution[target2] == "knight"
        else:
            target2 = None
            tg2_is_knight = False

        _parity_types = ("odd_knight", "even_knight", "odd_knave", "even_knave")
        available = [
            t for t in types
            if not (majority_pair_used and t in ("majority_knight", "majority_knave"))
            and not (at_least_knight_used and t == "at_least_knight")
            and not (at_least_knave_used and t == "at_least_knave")
            and not (parity_group_used and t in _parity_types)
            and not (if_then_used and t == "if_then")
            and not (target2 is None and t in _TWO_TARGET_TYPES_SET)
        ]
        random.shuffle(available)

        stmt = None
        for t in available:
            if t in _COUNTING_TYPES_SET:
                stmt = _make_counting_statement(t, speaker, sp_is_knight, knight_count, num_persons)
                if stmt is not None:
                    break
                stmt = None
            elif t in _TWO_TARGET_TYPES_SET:
                stmt = _make_two_target_statement(t, speaker, target, target2, sp_is_knight, tg_is_knight, tg2_is_knight)
                if stmt is not None:
                    break
                stmt = None
            elif _is_type_valid(t, sp_is_knight, tg_is_knight):
                stmt = _make_statement(t, speaker, target, tg_is_knight)
                break

        if stmt is None:
            # フォールバック: accusation / affirmation は必ず成立する
            t = "accusation" if sp_is_knight != tg_is_knight else "affirmation"
            stmt = _make_statement(t, speaker, target, tg_is_knight)

        used_type = stmt["type"]
        if used_type in ("majority_knight", "majority_knave"):
            majority_pair_used = True
        elif used_type == "at_least_knight":
            at_least_knight_used = True
        elif used_type == "at_least_knave":
            at_least_knave_used = True
        elif used_type in ("odd_knight", "even_knight", "odd_knave", "even_knave"):
            parity_group_used = True
        elif used_type == "if_then":
            if_then_used = True

        statements.append(stmt)

    return statements


def _is_unique_solution(puzzle: dict) -> bool:
    """z3でパズルの解が一意かどうか確認する."""
    persons = puzzle["persons"]
    solution = puzzle["solution"]

    is_knight = {p: Bool(f"is_knight_{p}") for p in persons}
    import z3
    solver_inst = z3.Solver()

    for stmt in puzzle["statements"]:
        add_constraint(solver_inst, stmt, is_knight)

    # 既知解の否定を追加 → 別解が存在しなければ unsat
    known = And([is_knight[p] if solution[p] == "knight" else Not(is_knight[p]) for p in persons])
    solver_inst.add(Not(known))

    return solver_inst.check() != sat


def generate_puzzle(num_persons: int, level: str, max_retries: int = 1000) -> dict:
    """パズルを生成して返す.

    Args:
        num_persons: 人数（2以上）
        level: 難易度（"easy" | "tricky" | "trickier"）
        max_retries: 最大リトライ回数

    Returns:
        生成されたパズル辞書（version, level, num_persons, persons, statements, solution を含む）

    Raises:
        ValueError: num_persons が 2 未満、または level が不正な場合
        RuntimeError: リトライ上限を超えた場合
    """
    if num_persons < 2:
        raise ValueError("num_persons は 2 以上が必要です")

    persons = [f"{chr(ord('A') + i)}さん" for i in range(num_persons)]

    if level == "easy":
        types = STATEMENT_TYPES_EASY
        num_statements = num_persons
    elif level == "tricky":
        types = STATEMENT_TYPES_ALL
        num_statements = num_persons
    elif level == "trickier":
        types = STATEMENT_TYPES_ALL
        num_statements = num_persons + num_persons // 2
    else:
        raise ValueError(f"不明な難易度: {level}")

    for _ in range(max_retries):
        solution = {p: random.choice(["knight", "knave"]) for p in persons}
        statements = _generate_statements(persons, solution, types, num_statements)

        puzzle = {
            "version": "1.0",
            "level": level,
            "num_persons": num_persons,
            "persons": persons,
            "statements": statements,
            "solution": solution,
        }

        if _is_unique_solution(puzzle):
            return puzzle

    raise RuntimeError(f"{max_retries} 回リトライしても一意なパズルを生成できませんでした")
