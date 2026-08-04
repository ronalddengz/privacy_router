#!/usr/bin/env python3
"""
RAT-Benchmark Adapter for Privacy Router Benchmark
===================================================
Loads the RAT-Benchmark dataset from HuggingFace and converts it to our
benchmark input format for testing the privacy-preserving text router.

RAT-Benchmark: Re-identification Attack Testing benchmark
- HuggingFace: imperial-cpg/rat-bench
- Paper: arXiv:2602.12806

Dataset Schema (discovered):
- id: int64 - Sample identifier
- profile: JSON - Full profile with all identifiers
- direct_identifiers: JSON - Direct PII (email, etc.)
- indirect_identifiers: JSON - Quasi-identifiers (DOB, SEX, ST, MAR, CIT)
- features: List[str] - Feature codes present in this sample
- difficulty: int64 - Re-identification difficulty level (1, 2, 3)
- prompt: str - The prompt used to generate the text
- text: str - The generated consultation transcript
- scenario: str - Scenario type (e.g., "Medical consultation")

Configs available: english, serbian, spanish, dutch, public information utility
"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
import argparse

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("WARNING: 'datasets' library not installed. Run: pip install datasets")


@dataclass
class RATSample:
    """A sample from the RAT-Benchmark dataset."""
    id: str
    text: str
    difficulty: int  # 1=easy to re-identify, 2=medium, 3=hard
    scenario: str
    direct_identifiers: Dict[str, Any]
    indirect_identifiers: Dict[str, Any]
    features: List[str]
    profile: Dict[str, Any]
    expected_critical: bool  # Our mapping: ALL RAT samples are sensitive by design


def explore_dataset(config: str = "english"):
    """
    Explore the RAT-Benchmark dataset structure.
    """
    if not HF_AVAILABLE:
        print("Cannot explore: 'datasets' library not available")
        return None
    
    print(f"Loading RAT-Benchmark config: {config}")
    
    try:
        dataset = load_dataset("imperial-cpg/rat-bench", config)
        
        print("\n" + "=" * 70)
        print("DATASET STRUCTURE")
        print("=" * 70)
        print(f"\nAvailable splits: {list(dataset.keys())}")
        
        for split_name, split_data in dataset.items():
            print(f"\n--- Split: {split_name} ---")
            print(f"  Number of samples: {len(split_data)}")
            print(f"  Column names: {split_data.column_names}")
            
            # Show difficulty distribution
            difficulties = {}
            for sample in split_data:
                d = sample.get('difficulty', 'unknown')
                difficulties[d] = difficulties.get(d, 0) + 1
            print(f"  Difficulty distribution: {difficulties}")
            
            # Show scenario distribution
            scenarios = {}
            for sample in split_data:
                s = sample.get('scenario', 'unknown')
                scenarios[s] = scenarios.get(s, 0) + 1
            print(f"  Scenario distribution: {scenarios}")
            
            # Show first sample in detail
            if len(split_data) > 0:
                print(f"\n  First sample details:")
                sample = split_data[0]
                print(f"    id: {sample['id']}")
                print(f"    difficulty: {sample['difficulty']}")
                print(f"    scenario: {sample['scenario']}")
                print(f"    direct_identifiers: {sample['direct_identifiers']}")
                print(f"    indirect_identifiers: {sample['indirect_identifiers']}")
                print(f"    features: {sample['features'][:10]}..." if len(sample['features']) > 10 else f"    features: {sample['features']}")
                
                text_preview = sample['text'][:500] + "..." if len(sample['text']) > 500 else sample['text']
                print(f"    text preview:\n{text_preview}")
        
        return dataset
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def convert_rat_to_benchmark(
    dataset,
    split: str = "train",
    max_samples: Optional[int] = None,
    difficulty_filter: Optional[List[int]] = None,
) -> List[RATSample]:
    """
    Convert RAT-Benchmark samples to our benchmark format.
    
    Args:
        dataset: HuggingFace dataset object
        split: Which split to use
        max_samples: Limit number of samples
        difficulty_filter: Only include samples with these difficulty levels [1,2,3]
    
    Returns:
        List of RATSample objects
    
    Note on expected_critical:
        ALL RAT-Benchmark samples are designed to contain re-identifiable information.
        The 'difficulty' level indicates how hard it is to re-identify, not whether
        the content is sensitive. Therefore, we mark ALL samples as expected_critical=True.
        
        - difficulty=1: Easy to re-identify (most direct identifiers exposed)
        - difficulty=2: Medium difficulty
        - difficulty=3: Hard to re-identify (more indirect identifiers)
        
        A well-functioning privacy system should flag ALL of these as sensitive,
        though it might route them differently based on risk level.
    """
    if split not in dataset:
        available = list(dataset.keys())
        print(f"Split '{split}' not found. Available: {available}")
        split = available[0] if available else None
        if not split:
            return []
        print(f"Using split: {split}")
    
    split_data = dataset[split]
    
    # Convert samples
    samples = []
    
    for i, row in enumerate(split_data):
        if max_samples and len(samples) >= max_samples:
            break
        
        difficulty = row.get('difficulty', 0)
        
        # Apply difficulty filter if specified
        if difficulty_filter and difficulty not in difficulty_filter:
            continue
        
        text = row.get('text', "")
        if not text or not isinstance(text, str):
            continue
        
        # ALL RAT samples are sensitive - they contain re-identifiable info by design
        # The difficulty just indicates HOW HARD it is to re-identify
        expected_critical = True
        
        samples.append(RATSample(
            id=str(row.get('id', i)),
            text=text,
            difficulty=difficulty,
            scenario=row.get('scenario', 'unknown'),
            direct_identifiers=row.get('direct_identifiers', {}),
            indirect_identifiers=row.get('indirect_identifiers', {}),
            features=row.get('features', []),
            profile=row.get('profile', {}),
            expected_critical=expected_critical,
        ))
    
    return samples


def export_to_benchmark_format(
    samples: List[RATSample],
    output_path: str,
) -> None:
    """
    Export RAT samples to the benchmark input format used by benchmark_revised.py
    
    Format:
    ---
    # name: <sample_name>
    # expected_critical: true/false
    
    <text content>
    
    ---
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write("---\n")
            f.write(f"# name: RAT-{sample.id} (difficulty={sample.difficulty}, {sample.scenario})\n")
            f.write(f"# expected_critical: {str(sample.expected_critical).lower()}\n")
            f.write(f"# rat_difficulty: {sample.difficulty}\n")
            f.write(f"# rat_scenario: {sample.scenario}\n")
            f.write(f"# rat_direct_ids: {list(sample.direct_identifiers.keys())}\n")
            f.write(f"# rat_indirect_ids: {list(sample.indirect_identifiers.keys())}\n")
            f.write("\n")
            f.write(sample.text)
            f.write("\n\n")
    
    print(f"\nExported {len(samples)} samples to: {output_path}")


def export_to_json(
    samples: List[RATSample],
    output_path: str,
) -> None:
    """Export samples to JSON for detailed analysis."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = []
    for s in samples:
        data.append({
            "id": s.id,
            "text": s.text,
            "difficulty": s.difficulty,
            "scenario": s.scenario,
            "expected_critical": s.expected_critical,
            "direct_identifiers": s.direct_identifiers,
            "indirect_identifiers": s.indirect_identifiers,
            "features": s.features,
            "profile": s.profile,
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Exported JSON to: {output_path}")


def print_statistics(samples: List[RATSample]) -> None:
    """Print statistics about the converted samples."""
    print("\n" + "=" * 70)
    print("CONVERSION STATISTICS")
    print("=" * 70)
    
    total = len(samples)
    if total == 0:
        print("No samples to analyze!")
        return
    
    critical = sum(1 for s in samples if s.expected_critical)
    not_critical = total - critical
    
    print(f"\nTotal samples: {total}")
    print(f"  Expected critical (sensitive): {critical} ({100*critical/total:.1f}%)")
    print(f"  Expected not critical (safe): {not_critical} ({100*not_critical/total:.1f}%)")
    
    # Difficulty distribution
    print(f"\nDifficulty distribution:")
    difficulty_counts = {}
    for s in samples:
        difficulty_counts[s.difficulty] = difficulty_counts.get(s.difficulty, 0) + 1
    for d in sorted(difficulty_counts.keys()):
        count = difficulty_counts[d]
        desc = {1: "Easy to re-identify", 2: "Medium", 3: "Hard to re-identify"}.get(d, "Unknown")
        print(f"  Level {d} ({desc}): {count} ({100*count/total:.1f}%)")
    
    # Scenario distribution
    print(f"\nScenario distribution:")
    scenario_counts = {}
    for s in samples:
        scenario_counts[s.scenario] = scenario_counts.get(s.scenario, 0) + 1
    for scenario, count in sorted(scenario_counts.items(), key=lambda x: -x[1]):
        print(f"  {scenario}: {count} ({100*count/total:.1f}%)")
    
    # Direct identifier types
    print(f"\nDirect identifier types found:")
    direct_id_types = {}
    for s in samples:
        for id_type in s.direct_identifiers.keys():
            direct_id_types[id_type] = direct_id_types.get(id_type, 0) + 1
    for id_type, count in sorted(direct_id_types.items(), key=lambda x: -x[1]):
        print(f"  {id_type}: {count} ({100*count/total:.1f}%)")
    
    # Indirect identifier types
    print(f"\nIndirect identifier types found:")
    indirect_id_types = {}
    for s in samples:
        for id_type in s.indirect_identifiers.keys():
            indirect_id_types[id_type] = indirect_id_types.get(id_type, 0) + 1
    for id_type, count in sorted(indirect_id_types.items(), key=lambda x: -x[1]):
        print(f"  {id_type}: {count} ({100*count/total:.1f}%)")
    
    # Text length statistics
    lengths = [len(s.text) for s in samples]
    print(f"\nText length statistics:")
    print(f"  Min: {min(lengths)} chars")
    print(f"  Max: {max(lengths)} chars")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f} chars")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]} chars")


def main():
    parser = argparse.ArgumentParser(
        description="Convert RAT-Benchmark to privacy router benchmark format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
RAT-Benchmark Information:
  All samples in RAT-Benchmark contain re-identifiable information by design.
  The 'difficulty' level (1-3) indicates how hard it is to re-identify someone,
  NOT whether the content is sensitive. Therefore:
  
  - ALL samples are marked as expected_critical=True
  - A good privacy system should flag all of them
  - Difficulty level can be used for stratified analysis
  
  Available configs: english, serbian, spanish, dutch, "public information utility"

Examples:
  # Explore dataset structure
  python rat_benchmark_adapter.py --explore
  
  # Convert English config (default)
  python rat_benchmark_adapter.py -o rat_benchmark_input.txt
  
  # Convert only difficulty level 1 (easiest to re-identify)
  python rat_benchmark_adapter.py -o rat_easy.txt --difficulty 1
  
  # Convert all difficulties, limit to 100 samples
  python rat_benchmark_adapter.py -o rat_100.txt -n 100
  
  # Then run benchmark
  python benchmark_revised.py rat_benchmark_input.txt -o rat_outputs/
        """
    )
    
    parser.add_argument(
        "--explore",
        action="store_true",
        help="Explore the dataset structure without converting"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="rat_benchmark_input.txt",
        help="Output file path (default: rat_benchmark_input.txt)"
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Also export to JSON file"
    )
    parser.add_argument(
        "-n", "--max-samples",
        type=int,
        default=None,
        help="Maximum samples to convert (default: all)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="english",
        choices=["english", "serbian", "spanish", "dutch", "public information utility"],
        help="Language/config to load (default: english)"
    )
    parser.add_argument(
        "--difficulty",
        type=int,
        nargs="+",
        default=None,
        help="Filter by difficulty level(s): 1, 2, and/or 3"
    )
    
    args = parser.parse_args()
    
    if not HF_AVAILABLE:
        print("ERROR: 'datasets' library required. Install with: pip install datasets")
        return 1
    
    if args.explore:
        explore_dataset(args.config)
        return 0
    
    # Load dataset
    print(f"Loading RAT-Benchmark (config={args.config})...")
    try:
        dataset = load_dataset("imperial-cpg/rat-bench", args.config)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return 1
    
    # Convert samples
    print(f"\nConverting samples from split: {args.split}")
    if args.difficulty:
        print(f"Filtering by difficulty levels: {args.difficulty}")
    
    samples = convert_rat_to_benchmark(
        dataset,
        split=args.split,
        max_samples=args.max_samples,
        difficulty_filter=args.difficulty,
    )
    
    if not samples:
        print("No samples converted!")
        return 1
    
    # Print statistics
    print_statistics(samples)
    
    # Export
    export_to_benchmark_format(samples, args.output)
    
    if args.json:
        export_to_json(samples, args.json)
    
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print(f"""
Run the benchmark on the converted dataset:

  python benchmark_revised.py {args.output} -o rat_benchmark_outputs/

Or with specific k value:

  python benchmark_revised.py {args.output} -o rat_benchmark_outputs/ -k 5

Key metrics to watch:
  - False Cloud Release Rate: Should be LOW (we want to catch all sensitive content)
  - Tier Distribution: Most RAT samples should route to TIER_2 or TIER_3
  - By Difficulty: Level 1 should be easier to detect than Level 3
""")
    
    return 0


if __name__ == "__main__":
    exit(main())