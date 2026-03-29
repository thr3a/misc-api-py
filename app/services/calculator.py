"""簡単な計算ユーティリティ。"""

from __future__ import annotations


def add(a: int | float, b: int | float) -> int | float:
    """2つの数値を足し算する。

    Args:
        a: 1つ目の数値。
        b: 2つ目の数値。

    Returns:
        足し算の結果。
    """
    return a + b


def subtract(a: int | float, b: int | float) -> int | float:
    """2つの数値を引き算する。

    Args:
        a: 1つ目の数値（被減数）。
        b: 2つ目の数値（減数）。

    Returns:
        引き算の結果。
    """
    return a - b


def multiply(a: int | float, b: int | float) -> int | float:
    """2つの数値を掛け算する。

    Args:
        a: 1つ目の数値。
        b: 2つ目の数値。

    Returns:
        掛け算の結果。
    """
    return a * b


def divide(a: int | float, b: int | float) -> float:
    """2つの数値を割り算する。

    Args:
        a: 1つ目の数値（被除数）。
        b: 2つ目の数値（除数）。

    Returns:
        割り算の結果。

    Raises:
        ZeroDivisionError: 除数が0の場合。
    """
    if b == 0:
        msg = "除数は0以外である必要があります。"
        raise ZeroDivisionError(msg)
    return a / b


def is_even(n: int) -> bool:
    """整数が偶数かどうかを判定する。

    Args:
        n: 判定する整数。

    Returns:
        偶数の場合はTrue、奇数の場合はFalse。
    """
    return n % 2 == 0


def factorial(n: int) -> int:
    """非負整数の階乗を計算する。

    Args:
        n: 非負整数。

    Returns:
        nの階乗。

    Raises:
        ValueError: nが負の数の場合。
    """
    if n < 0:
        msg = "nは0以上の整数である必要があります。"
        raise ValueError(msg)
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result
