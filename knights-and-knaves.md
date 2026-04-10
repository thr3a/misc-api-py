# Knights and Knaves パズル

## パズルのルール

- 全員がKnight（真実のみ）かKnave（嘘のみ）
- 発言から誰がどちらかを論理的に推論する

## 発言タイプとz3制約

### ターゲット型（1ターゲット）

| type | 意味 | z3制約 |
|---|---|---|
| accusation | 「Xは悪党だ」 | `is_knight[sp] != is_knight[tg]` |
| affirmation | 「Xは騎士だ」 | `is_knight[sp] == is_knight[tg]` |
| sympathetic | 「Xは私と同じ種類だ」 | `is_knight[tg] == True` |
| antithetic | 「Xは私と異なる種類だ」 | `is_knight[tg] == False` |
| disjoint | 「Xは騎士/悪党だ、または私は悪党だ」 | `is_knight[sp] == True`, `is_knight[tg] == claimed_is_knight` |
| joint | 「Xは悪党/騎士だ、そして私は悪党だ」 | `is_knight[sp] == False`, `is_knight[tg] == not claimed_is_knight` |

disjoint/joint は `claimed_is_knight`（bool）をJSONに保存する。

### 2ターゲット型（2人に言及）

speaker 以外の2人（`target`, `target2`）が必要。**num_persons >= 3 のときのみ使用可能**。

| type | 意味 | z3制約 | 出現制限 |
|---|---|---|---|
| pair_same | 「XとYは同じ種類だ」 | `is_knight[sp] == (is_knight[tg] == is_knight[tg2])` | なし |
| pair_diff | 「XとYは異なる種類だ」 | `is_knight[sp] == (is_knight[tg] != is_knight[tg2])` | なし |
| if_then | 「もしXが騎士/悪党なら、Yは騎士/悪党だ」 | `is_knight[sp] == Implies(is_knight[tg] == BoolVal(ck), is_knight[tg2] == BoolVal(ek))` | 1回/パズル |
| either_knight | 「XかYの少なくとも一方は騎士だ」 | `is_knight[sp] == Or(is_knight[tg], is_knight[tg2])` | なし |
| either_knave | 「XかYの少なくとも一方は悪党だ」 | `is_knight[sp] == Or(Not(is_knight[tg]), Not(is_knight[tg2]))` | なし |

`if_then` のJSONフィールド:
- `condition_knight`（bool）: 前件が「騎士なら」(True)か「悪党なら」(False)か
- `conclusion_knight`（bool）: 後件が「騎士だ」(True)か「悪党だ」(False)か
- 4パターンの組み合わせからランダムに選択（解と整合するものを採用）

#### 2ターゲット型の成立条件

| type | 成立条件 |
|---|---|
| pair_same | `sp_is_knight == (tg1_is_knight == tg2_is_knight)` |
| pair_diff | `sp_is_knight == (tg1_is_knight != tg2_is_knight)` |
| if_then | `antecedent = (tg1_is_knight == ck)`, `consequent = (tg2_is_knight == ek)`, `sp_is_knight == ((not antecedent) or consequent)` |
| either_knight | `sp_is_knight == (tg1_is_knight or tg2_is_knight)` |
| either_knave | `sp_is_knight == ((not tg1_is_knight) or (not tg2_is_knight))` |

### カウント型（全員の構成に言及）

speaker の真偽が発言の真偽に反映される（騎士なら真、悪党なら偽として発言）。  
`knight_count = Sum([If(is_knight[p], 1, 0) for p in persons])`

| type | 意味 | z3制約 | JSONフィールド |
|---|---|---|---|
| at_least_knight | 「騎士は少なくともN人いる」 | `is_knight[sp] == (knight_count >= n)` | `n` |
| majority_knight | 「騎士のほうが多い」 | `is_knight[sp] == (knight_count * 2 > num_persons)` | なし |
| at_least_knave | 「悪党は少なくともN人いる」 | `is_knight[sp] == (knave_count >= n)` | `n` |
| majority_knave | 「悪党のほうが多い」 | `is_knight[sp] == (knave_count * 2 > num_persons)` | なし |
| odd_knight | 「この中に騎士は奇数人いる」 | `is_knight[sp] == (knight_count % 2 == 1)` | なし |
| even_knight | 「この中に騎士は偶数人いる」 | `is_knight[sp] == (knight_count % 2 == 0)` | なし |
| odd_knave | 「この中に悪党は奇数人いる」 | `is_knight[sp] == (knave_count % 2 == 1)` | なし |
| even_knave | 「この中に悪党は偶数人いる」 | `is_knight[sp] == (knave_count % 2 == 0)` | なし |

`knave_count = num_persons - knight_count`

#### カウント型の組み合わせ制約

| 組み合わせ | 問題 | 備考 |
|---|---|---|
| `majority_knight` + `majority_knave` が両方騎士から | 同時に真にはなれないため矛盾 | `num_persons >= 2` の場合は必ず片方が偽 |
| `at_least_knight(0)` を騎士が発言 | 「0人以上」は恒真のため自明 | 禁止推奨（生成時にスキップ） |
| `at_least_knave(0)` を騎士が発言 | 同上 | 禁止推奨（生成時にスキップ） |
| `at_least_knight(N)` で N > num_persons を騎士が発言 | 恒偽の発言を騎士がする → 矛盾 | 生成時にスキップ |
| `at_least_knight/knave` の `n` が大きすぎる | 不自然な文面になりやすい | `n <= min(MAX_VISIBLE_COUNT_N, num_persons)` に制限し、候補が空なら別タイプへフォールバック |
| `odd_knight` / `even_knight` / `odd_knave` / `even_knave` | 奇偶性パリティ系は1クイズ中に合計1回まで（4タイプをまとめて `parity_group` として管理） | `parity_group_used` フラグで制御 |
| `odd_knight` と `even_knight` を同一クイズで両方使用 | 互いに否定関係のため同話者が発言すると矛盾、異話者でも冗長 | `parity_group_used` フラグで防止 |

#### 出現制限フラグ一覧

| フラグ | 対象タイプ | 上限 |
|---|---|---|
| `majority_pair_used` | `majority_knight`, `majority_knave` | 合計1回 |
| `at_least_knight_used` | `at_least_knight` | 1回 |
| `at_least_knave_used` | `at_least_knave` | 1回 |
| `parity_group_used` | `odd_knight`, `even_knight`, `odd_knave`, `even_knave` | 合計1回 |
| `if_then_used` | `if_then` | 1回 |

## JSONスキーマ

`text` フィールドは不要。`type`/`speaker`/`target`/`target2`/`claimed_is_knight`/`condition_knight`/`conclusion_knight`/`n` から puzzle_lib.py の `to_text()` で動的生成する。

```json
{
  "version": "1.0",
  "level": "trickier",
  "num_persons": 3,
  "persons": ["Aさん", "Bさん", "Cさん"],
  "statements": [
    {
      "type": "accusation",
      "speaker": "Aさん",
      "target": "Bさん"
    },
    {
      "type": "pair_same",
      "speaker": "Bさん",
      "target": "Aさん",
      "target2": "Cさん"
    },
    {
      "type": "if_then",
      "speaker": "Cさん",
      "target": "Aさん",
      "target2": "Bさん",
      "condition_knight": true,
      "conclusion_knight": false
    },
    {
      "type": "majority_knave",
      "speaker": "Aさん"
    }
  ],
  "solution": {
    "Aさん": "knight",
    "Bさん": "knave",
    "Cさん": "knight"
  }
}
```

フィールドの必要性:
- `target`: ターゲット型・2ターゲット型のみ必要
- `target2`: 2ターゲット型のみ必要
- `claimed_is_knight`: disjoint/joint のみ必要
- `condition_knight`, `conclusion_knight`: if_then のみ必要
- `n`: at_least_knight/at_least_knave のみ必要
- カウント型は `target` フィールド不要

## 難易度設定

| level | 使用タイプ | ステートメント数 |
|---|---|---|
| easy | accusation, affirmation, sympathetic, antithetic | `num_persons` 本 |
| normal | 全19種類 | `num_persons` 本 |
| hard | 全19種類 | `num_persons + num_persons // 2` 本 |

hard では各personが複数回speakerになり得る。

## ファイル構成

```
/home/thr3a/work/puzzle-py/
├── generate.py   # 問題生成
├── answer.py     # 自動解答
└── puzzle_lib.py # 共通ロジック
```

### ステートメント生成ロジック

1. 各personをspeakerとして1つ発言を生成（全員がグラフに含まれるよう）
2. target はspeaker以外からランダム選択
3. 2ターゲット型の場合は target2 を target 以外のspeaker以外からランダム選択（num_persons >= 3 のときのみ候補に含める）
4. 難易度に応じたタイプをランダム選択し、論理的に成立するものを採用
5. タイプが成立しない場合（型の組み合わせ不一致）はaccusation/affirmationにフォールバック

`at_least_knight` / `at_least_knave` の `n` は見た目の自然さのため `MAX_VISIBLE_COUNT_N` 以下に制限する。  
上限適用後に候補が空になる場合、その count 系タイプは不採用として他タイプを試す。

### タイプの成立条件（ターゲット型）

| type | 成立条件 |
|---|---|
| accusation | `solution[sp] != solution[tg]` |
| affirmation | `solution[sp] == solution[tg]` |
| sympathetic | `solution[tg] == True` |
| antithetic | `solution[tg] == False` |
| disjoint | `solution[sp] == True`（Knightのみ） |
| joint | `solution[sp] == False`（Knaveのみ） |

## 一意解の保証

JSは構造的に連結グラフ（`joinConnectedSets`）を作ることで解の一意性を担保している。
Python実装では生成後にz3で明示的に検証する：

1. ランダムな真偽割り当て（solution）を生成
2. solutionと整合するステートメントを生成
3. z3でsolution以外の解が存在しないか確認
4. 別解が存在する場合は1に戻り再生成（最大リトライ数を設ける）

## 検証方法

```bash
# 問題生成
python generate.py --num 3 --level easy --output q.json

# 自動解答（solution フィールドを見ずに解く）
python answer.py --input q.json
```
