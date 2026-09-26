import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]

RESULT_PATH = (
    ROOT_DIR
    / "data"
    / "evaluation"
    / "benchmark_results.json"
)


def main():
    with RESULT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        results = json.load(file)

    systems = results["systems"]

    for system_name, system_data in systems.items():
        print("\n" + "=" * 80)
        print(f"{system_name.upper()} — FAILED QUERIES")
        print("=" * 80)

        failures = 0

        for item in system_data["per_query"]:
            relevant = set(item["relevant_ids"])
            retrieved = set(item["retrieved_ids"])

            if not relevant.intersection(retrieved):
                failures += 1

                print(f"\n[{item['id']}]")
                print(f"Query: {item['query']}")
                print(f"Relevant: {item['relevant_ids']}")
                print(f"Retrieved: {item['retrieved_ids']}")

        if failures == 0:
            print("\nNo complete retrieval failures.")

        print(f"\nTotal complete failures: {failures}")


if __name__ == "__main__":
    main()