#!/usr/bin/env python3
"""
Script to map sentence IDs to original sentences for manual validation.

This script reads the request body (with original sentences) and response body
(with sentence IDs), then creates an enhanced output JSON file with original
sentences included for easy manual review.

Supports two output modes:
  - Full mode (default): Includes both IDs and sentences for validation
  - Concise mode (--concise): LLM-friendly format with only sentence text, no IDs

Usage:
    python map_sentences.py <request_file> <response_file> <output_file> [--concise]

Examples:
    # Full output with IDs
    python map_sentences.py data/input_example.json test_output_standalone.json output_mapped.json

    # Concise LLM-friendly output
    python map_sentences.py data/input_example.json test_output_standalone.json output_llm.json --concise
"""

import sys
import json
import argparse
from typing import Dict, List, Any


def create_sentence_map(request_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Create a mapping of sentence ID to original sentence text from request data.

    Args:
        request_data: The request JSON data containing baseline (and optionally comparison)

    Returns:
        Dictionary mapping sentence_id -> sentence_text
    """
    sentence_map = {}

    # Process baseline sentences
    if 'baseline' in request_data:
        for item in request_data['baseline']:
            sentence_id = item.get('id')
            sentence_text = item.get('sentence')
            if sentence_id and sentence_text:
                # Keep first occurrence if duplicate IDs exist
                if sentence_id not in sentence_map:
                    sentence_map[sentence_id] = sentence_text

    # Process comparison sentences (for comparative analysis)
    if 'comparison' in request_data:
        for item in request_data['comparison']:
            sentence_id = item.get('id')
            sentence_text = item.get('sentence')
            if sentence_id and sentence_text:
                if sentence_id not in sentence_map:
                    sentence_map[sentence_id] = sentence_text

    return sentence_map


def map_standalone_response(
    response_data: Dict[str, Any],
    sentence_map: Dict[str, str],
    concise: bool = False
) -> Dict[str, Any]:
    """
    Enhance standalone analysis response with original sentences.

    Args:
        response_data: The response JSON data
        sentence_map: Mapping of ID to sentence text
        concise: If True, create LLM-friendly output (exclude IDs, sentences field)

    Returns:
        Enhanced response with original sentences
    """
    enhanced_response = response_data.copy()

    if 'clusters' in enhanced_response:
        for cluster in enhanced_response['clusters']:
            # Add mapped sentences field
            sentence_ids = cluster.get('sentences', [])

            if concise:
                # LLM-friendly format: only sentence text, no IDs
                cluster['mapped_sentences'] = [
                    sentence_map.get(sid, f"[NOT FOUND: {sid}]")
                    for sid in sentence_ids
                ]
                # Remove the sentences field (ID list)
                cluster.pop('sentences', None)
            else:
                # Full format: include both ID and sentence
                cluster['mapped_sentences'] = [
                    {
                        'id': sid,
                        'sentence': sentence_map.get(sid, f"[NOT FOUND: {sid}]")
                    }
                    for sid in sentence_ids
                ]

    return enhanced_response


def map_comparative_response(
    response_data: Dict[str, Any],
    sentence_map: Dict[str, str],
    concise: bool = False
) -> Dict[str, Any]:
    """
    Enhance comparative analysis response with original sentences.

    Args:
        response_data: The response JSON data
        sentence_map: Mapping of ID to sentence text
        concise: If True, create LLM-friendly output (exclude IDs, ID list fields)

    Returns:
        Enhanced response with original sentences
    """
    enhanced_response = response_data.copy()

    if 'clusters' in enhanced_response:
        for cluster in enhanced_response['clusters']:
            # Add mapped baseline sentences
            baseline_ids = cluster.get('baselineSentences', [])

            if concise:
                # LLM-friendly format: only sentence text, no IDs
                cluster['mapped_baseline_sentences'] = [
                    sentence_map.get(sid, f"[NOT FOUND: {sid}]")
                    for sid in baseline_ids
                ]
                # Remove the baselineSentences field (ID list)
                cluster.pop('baselineSentences', None)
            else:
                # Full format: include both ID and sentence
                cluster['mapped_baseline_sentences'] = [
                    {
                        'id': sid,
                        'sentence': sentence_map.get(sid, f"[NOT FOUND: {sid}]")
                    }
                    for sid in baseline_ids
                ]

            # Add mapped comparison sentences
            comparison_ids = cluster.get('comparisonSentences', [])

            if concise:
                # LLM-friendly format: only sentence text, no IDs
                cluster['mapped_comparison_sentences'] = [
                    sentence_map.get(sid, f"[NOT FOUND: {sid}]")
                    for sid in comparison_ids
                ]
                # Remove the comparisonSentences field (ID list)
                cluster.pop('comparisonSentences', None)
            else:
                # Full format: include both ID and sentence
                cluster['mapped_comparison_sentences'] = [
                    {
                        'id': sid,
                        'sentence': sentence_map.get(sid, f"[NOT FOUND: {sid}]")
                    }
                    for sid in comparison_ids
                ]

    return enhanced_response


def detect_analysis_type(response_data: Dict[str, Any]) -> str:
    """
    Detect whether response is from standalone or comparative analysis.

    Args:
        response_data: The response JSON data

    Returns:
        'standalone' or 'comparative'
    """
    if 'clusters' in response_data and response_data['clusters']:
        first_cluster = response_data['clusters'][0]
        if 'baselineSentences' in first_cluster or 'comparisonSentences' in first_cluster:
            return 'comparative'
    return 'standalone'


def main():
    """Main function to run the mapping script."""
    parser = argparse.ArgumentParser(
        description='Map sentence IDs to original sentences in analysis response',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standalone analysis (full output with IDs)
  python map_sentences.py data/input_example.json test_output_standalone.json output_mapped.json

  # Standalone analysis (concise LLM-friendly output)
  python map_sentences.py data/input_example.json test_output_standalone.json output_llm.json --concise

  # Comparative analysis (full output)
  python map_sentences.py data/input_comparison_example.json test_output_comparative.json output_mapped.json

  # Comparative analysis (concise LLM-friendly output)
  python map_sentences.py data/input_comparison_example.json test_output_comparative.json output_llm.json --concise
        """
    )

    parser.add_argument(
        'request_file',
        help='Path to request JSON file (input data with original sentences)'
    )
    parser.add_argument(
        'response_file',
        help='Path to response JSON file (output with sentence IDs)'
    )
    parser.add_argument(
        'output_file',
        help='Path to output JSON file (enhanced output with sentences)'
    )
    parser.add_argument(
        '--concise',
        action='store_true',
        help='Create concise LLM-friendly output (exclude IDs, keep only sentence text)'
    )

    args = parser.parse_args()

    # Read request file
    print(f"Reading request file: {args.request_file}")
    try:
        with open(args.request_file, 'r', encoding='utf-8') as f:
            request_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Request file not found: {args.request_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in request file: {e}")
        sys.exit(1)

    # Read response file
    print(f"Reading response file: {args.response_file}")
    try:
        with open(args.response_file, 'r', encoding='utf-8') as f:
            response_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Response file not found: {args.response_file}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in response file: {e}")
        sys.exit(1)

    # Create sentence mapping
    print("Creating sentence ID to text mapping...")
    sentence_map = create_sentence_map(request_data)
    print(f"Mapped {len(sentence_map)} unique sentence IDs")

    # Detect analysis type and map accordingly
    analysis_type = detect_analysis_type(response_data)
    print(f"Detected analysis type: {analysis_type}")

    if args.concise:
        print("Using concise mode (LLM-friendly output)")

    if analysis_type == 'comparative':
        enhanced_response = map_comparative_response(response_data, sentence_map, args.concise)
    else:
        enhanced_response = map_standalone_response(response_data, sentence_map, args.concise)

    # Write output file
    print(f"Writing enhanced output to: {args.output_file}")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(enhanced_response, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error: Failed to write output file: {e}")
        sys.exit(1)

    # Summary
    print("\n" + "="*60)
    print("Mapping complete!")
    print("="*60)
    if 'clusters' in enhanced_response:
        print(f"Total clusters: {len(enhanced_response['clusters'])}")
        for i, cluster in enumerate(enhanced_response['clusters'], 1):
            title = cluster.get('title', 'Unknown')
            if analysis_type == 'comparative':
                baseline_count = len(cluster.get('mapped_baseline_sentences', []))
                comparison_count = len(cluster.get('mapped_comparison_sentences', []))
                print(f"  Cluster {i}: '{title}' (baseline: {baseline_count}, comparison: {comparison_count})")
            else:
                sentence_count = len(cluster.get('mapped_sentences', []))
                print(f"  Cluster {i}: '{title}' ({sentence_count} sentences)")

    print(f"\nOutput saved to: {args.output_file}")


if __name__ == "__main__":
    main()
