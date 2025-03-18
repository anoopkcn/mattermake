import json
import re
import os
from collections import Counter
from typing import List, Dict


def parse_composition(slice_str: str) -> Counter:
    """Extract atomic composition from a SLICES string.

    The composition is the collection of atomic symbols at the beginning of the string.
    """
    # Extract all element symbols (capital letter followed by optional lowercase letter)
    elements = re.findall(r"[A-Z][a-z]?", slice_str)

    # Filter out any non-element looking patterns (typically digits or special chars)
    elements = [e for e in elements if not any(c.isdigit() for c in e)]

    return Counter(elements)


def extract_edges(slice_str: str) -> List[str]:
    """Extract edge representations from a SLICES string."""
    # Pattern for edges: digit + digit + (o|+|-) + (o|+|-) + (o|+|-)
    # This is a simplified pattern and might need adjustment based on your exact format
    edges = re.findall(r"\d+\d+[o+-][o+-][o+-]", slice_str)
    return edges


def calculate_metrics(generated_slices: List[str], original_slices: List[str]) -> Dict:
    """Calculate various performance metrics between generated and original slices."""
    # total_samples = len(generated_slices)
    metrics = {
        "composition_accuracy": 0.0,
        "exact_match": 0.0,
        "edge_accuracy": 0.0,
        "jaccard_similarity_avg": 0.0,
        "composition_recall_avg": 0.0,
        "composition_precision_avg": 0.0,
        "element_counts": {},
        "sample_metrics": [],
    }

    all_elements = set()
    element_hits = Counter()
    element_totals = Counter()

    for i, (gen, orig) in enumerate(zip(generated_slices, original_slices)):
        # Skip empty strings
        if not gen or not orig:
            continue

        # Extract compositions
        gen_comp = parse_composition(gen)
        orig_comp = parse_composition(orig)

        # Update element statistics
        for elem in orig_comp:
            all_elements.add(elem)
            element_totals[elem] += orig_comp[elem]
            element_hits[elem] += min(gen_comp[elem], orig_comp[elem])

        # Calculate composition similarity
        intersection = sum((gen_comp & orig_comp).values())
        union = sum((gen_comp | orig_comp).values())
        jaccard = intersection / union if union > 0 else 0

        # Calculate precision and recall for composition
        recall = (
            intersection / sum(orig_comp.values()) if sum(orig_comp.values()) > 0 else 0
        )
        precision = (
            intersection / sum(gen_comp.values()) if sum(gen_comp.values()) > 0 else 0
        )

        # Check if compositions match exactly
        comp_match = gen_comp == orig_comp

        # Check if the entire slice matches exactly
        exact_match = gen == orig

        # Extract and compare edges
        gen_edges = extract_edges(gen)
        orig_edges = extract_edges(orig)

        # Calculate edge accuracy
        edge_intersection = set(gen_edges) & set(orig_edges)
        edge_accuracy = len(edge_intersection) / len(orig_edges) if orig_edges else 1.0

        # Add metrics for this sample
        sample_metrics = {
            "id": i,
            "generated": gen,
            "original": orig,
            "composition_match": comp_match,
            "exact_match": exact_match,
            "jaccard_similarity": jaccard,
            "composition_recall": recall,
            "composition_precision": precision,
            "edge_accuracy": edge_accuracy,
        }
        metrics["sample_metrics"].append(sample_metrics)

        # Update aggregate metrics
        metrics["composition_accuracy"] += int(comp_match)
        metrics["exact_match"] += int(exact_match)
        metrics["edge_accuracy"] += edge_accuracy
        metrics["jaccard_similarity_avg"] += jaccard
        metrics["composition_recall_avg"] += recall
        metrics["composition_precision_avg"] += precision

    # Calculate per-element recall
    element_recall = {}
    for elem in all_elements:
        element_recall[elem] = (
            element_hits[elem] / element_totals[elem] if element_totals[elem] > 0 else 0
        )
    metrics["element_recall"] = element_recall

    # Calculate average metrics
    valid_samples = len(metrics["sample_metrics"])
    if valid_samples > 0:
        metrics["composition_accuracy"] /= valid_samples
        metrics["exact_match"] /= valid_samples
        metrics["edge_accuracy"] /= valid_samples
        metrics["jaccard_similarity_avg"] /= valid_samples
        metrics["composition_recall_avg"] /= valid_samples
        metrics["composition_precision_avg"] /= valid_samples

    # Add overall element statistics
    metrics["element_counts"] = {
        "total": dict(element_totals),
        "correct": dict(element_hits),
    }

    # Calculate F1 score
    if metrics["composition_precision_avg"] + metrics["composition_recall_avg"] > 0:
        metrics["composition_f1"] = (
            2 * metrics["composition_precision_avg"] * metrics["composition_recall_avg"]
        ) / (metrics["composition_precision_avg"] + metrics["composition_recall_avg"])
    else:
        metrics["composition_f1"] = 0.0

    return metrics


def main():
    # Load results from the generate.py output
    results_file = "../results/generation_results_cross_large_full.json"

    if not os.path.exists(results_file):
        print(f"Error: Results file {results_file} not found!")
        return

    with open(results_file, "r") as f:
        results = json.load(f)

    # Extract generated and original slices
    generated_slices = [r["generated"] for r in results]
    original_slices = [r["original"] for r in results]

    # Calculate metrics
    metrics = calculate_metrics(generated_slices, original_slices)

    # Print summary
    print("=" * 50)
    print("SLICE GENERATION EVALUATION METRICS")
    print("=" * 50)
    print(f"Number of samples: {len(metrics['sample_metrics'])}")
    print(f"Composition Accuracy: {metrics['composition_accuracy']:.4f}")
    print(f"Exact Match Accuracy: {metrics['exact_match']:.4f}")
    print(f"Edge Structure Accuracy: {metrics['edge_accuracy']:.4f}")
    print(f"Composition Jaccard Similarity: {metrics['jaccard_similarity_avg']:.4f}")
    print(f"Composition Recall: {metrics['composition_recall_avg']:.4f}")
    print(f"Composition Precision: {metrics['composition_precision_avg']:.4f}")
    print(f"Composition F1 Score: {metrics['composition_f1']:.4f}")

    print("\nPer-element Recall:")
    # Sort elements by recall
    sorted_elements = sorted(
        metrics["element_recall"].items(), key=lambda x: x[1], reverse=True
    )
    for elem, recall in sorted_elements:
        total = metrics["element_counts"]["total"][elem]
        correct = metrics["element_counts"]["correct"][elem]
        print(f"  {elem}: {recall:.4f} ({correct}/{total})")

    # Save detailed metrics to file
    output_metrics_file = "../results/evaluation_metrics_cross_large_full.json"
    with open(output_metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDetailed metrics saved to {output_metrics_file}")


if __name__ == "__main__":
    main()
