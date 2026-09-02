"""Check arithmetic and exact repetitions in Table 1 of the QAQ paper."""

from __future__ import annotations

from statistics import mean


QUALITY_FIELDS = ("hellaswag", "piqa", "arc_e", "arc_c", "winogrande", "wt2", "ptb")


TABLE = {
    "LLaMA3.1-8B": {
        "fp16": [78.90, 81.18, 81.10, 53.50, 73.48, 6.24, 9.01, 75.12, 23.12],
        "static8": [59.99, 79.98, 81.65, 51.45, 73.56, 6.24, 9.01, 56.61, 13.26],
        "static4": [59.29, 80.30, 81.44, 50.09, 72.93, 6.71, 9.10, 55.72, 13.26],
        "qaq_off": [59.99, 79.98, 81.65, 51.45, 73.56, 6.24, 9.01, 53.23, 13.26],
        "qaq_on": [59.99, 79.98, 81.65, 51.45, 73.56, 6.24, 9.01, 80.21, 12.52],
    },
    "Qwen3-4B": {
        "fp16": [68.42, 74.97, 78.49, 53.92, 66.06, 13.64, 18.74, 44.23, 14.23],
        "static8": [68.45, 74.81, 78.32, 53.92, 65.98, 14.83, 18.75, 38.91, 9.25],
        "static4": [66.78, 75.14, 76.94, 52.99, 63.22, 14.83, 20.25, 38.04, 9.25],
        "qaq_off": [68.45, 74.81, 78.32, 53.92, 65.98, 14.85, 18.75, 37.22, 9.25],
        "qaq_on": [68.45, 74.81, 78.32, 53.92, 65.98, 14.85, 18.75, 55.09, 8.78],
    },
    "Qwen3-8B": {
        "fp16": [74.95, 77.80, 80.89, 56.48, 67.72, 9.72, 13.54, 72.32, 19.32],
        "static8": [74.92, 77.74, 80.85, 56.57, 67.80, 10.30, 13.54, 65.64, 11.42],
        "static4": [73.98, 77.80, 80.18, 57.42, 66.06, 10.30, 14.41, 64.91, 11.42],
        "qaq_off": [74.92, 77.74, 80.85, 56.57, 67.80, 10.30, 13.54, 61.99, 11.42],
        "qaq_on": [74.92, 77.74, 80.85, 56.57, 67.80, 10.30, 13.54, 93.12, 10.80],
    },
}


def percent_change(new: float, baseline: float) -> float:
    return 100.0 * (new - baseline) / baseline


def quality(row: list[float]) -> tuple[float, ...]:
    return tuple(row[: len(QUALITY_FIELDS)])


def main() -> None:
    latency_reductions = []
    latency_overheads = []
    memory_savings = []

    print("Table 1 arithmetic audit")
    print()
    print("| Model | QAQ off vs static 4 latency | QAQ on vs static 8 latency | QAQ on memory saving |")
    print("| --- | ---: | ---: | ---: |")

    for model, rows in TABLE.items():
        if quality(rows["qaq_off"]) != quality(rows["static8"]):
            differing = [
                name
                for name, left, right in zip(
                    QUALITY_FIELDS, quality(rows["qaq_off"]), quality(rows["static8"])
                )
                if left != right
            ]
            # Qwen3-4B differs by 0.02 on WT2 in the printed table. Keep that
            # visible rather than weakening the comparison silently.
            if model != "Qwen3-4B" or differing != ["wt2"]:
                raise AssertionError(f"unexpected QAQ-off differences for {model}: {differing}")

        if quality(rows["qaq_on"]) != quality(rows["qaq_off"]):
            raise AssertionError(f"on-demand mode changes printed quality for {model}")
        if rows["static8"][8] != rows["static4"][8] or rows["qaq_off"][8] != rows["static8"][8]:
            raise AssertionError(f"printed static/off memory identity changed for {model}")

        off_vs_static4 = -percent_change(rows["qaq_off"][7], rows["static4"][7])
        on_vs_static8 = percent_change(rows["qaq_on"][7], rows["static8"][7])
        memory_saved = -percent_change(rows["qaq_on"][8], rows["static8"][8])
        latency_reductions.append(off_vs_static4)
        latency_overheads.append(on_vs_static8)
        memory_savings.append(memory_saved)

        print(
            f"| {model} | {off_vs_static4:.2f}% faster | "
            f"{on_vs_static8:.2f}% slower | {memory_saved:.2f}% |"
        )

    print()
    print(f"Simple mean QAQ-off latency reduction vs static 4-bit: {mean(latency_reductions):.2f}%")
    print(f"Simple mean on-demand latency overhead vs static 8-bit: {mean(latency_overheads):.2f}%")
    print(f"Simple mean on-demand memory saving vs static 8-bit: {mean(memory_savings):.2f}%")
    print()
    print("Printed-quality observations:")
    print("- On-demand on and off rows are identical for all seven quality fields in every model.")
    print("- QAQ off matches static 8-bit exactly except Qwen3-4B WT2: 14.85 versus 14.83.")
    print("- Static 4-bit, static 8-bit, and QAQ off have identical printed memory within each model.")
    print("- The 41.7% latency summary matches the mean overhead relative to static 8-bit.")
    print("- The claimed 4.5% lower latency holds for two rows; the three-model simple mean is 3.71%.")


if __name__ == "__main__":
    main()

