"""Standard text-entry metrics used in HCI research.

* WPM      : (|T| - 1) / seconds * 60 / 5        (MacKenzie)
* MSD error: MSD(P, T) / max(|P|, |T|) * 100      (Soukoreff & MacKenzie 2003)
* KSPC     : input symbols / |T|                  (MacKenzie 2002); here a
             "keystroke" is one dot or dash, so it measures Morse efficiency.
"""

from __future__ import annotations


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def wpm(transcribed: str, seconds: float) -> float:
    if seconds <= 0:
        return 0.0
    return max(len(transcribed) - 1, 0) / seconds * 60 / 5


def msd_error_rate(presented: str, transcribed: str) -> float:
    longest = max(len(presented), len(transcribed))
    return 100.0 * levenshtein(presented, transcribed) / longest if longest else 0.0


def kspc(symbols: int, transcribed: str) -> float:
    return symbols / len(transcribed) if transcribed else 0.0
