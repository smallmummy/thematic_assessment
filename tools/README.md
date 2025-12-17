# Tools Guide

## Map Sentences

The `map_sentences.py` script helps you map sentence IDs in the API response back to their original sentence text for manual validation and review.

## Usage

### Basic Command

```bash
python map_sentences.py <request_file> <response_file> <output_file> [--concise]
```

`--conise` flag will remove the field "sentences" and sub field "id" under "mapped_sentences", the output by this flag will be easy to feed to LLM to do the further analysis

### Example 

```bash
# After running test_standalone.py, you have:
# - data/input_example.json (request)
# - test_output_standalone.json (response)

python map_sentences.py \
    data/input_example.json \
    test_output_standalone.json \
    output_mapped_standalone.json
```

**Output:** `output_mapped_standalone.json` will contain:
```json
{
  "clusters": [
    {
      "title": "Withdrawal Issues",
      "sentiment": "negative",
      "sentences": ["7d4aa701-f1b2-41eb", "bc69d50c-4bdc-4fea"],
      "keyInsights": [...],
      "mapped_sentences": [
        {
          "id": "7d4aa701-f1b2-41eb",
          "sentence": "Withholding my money"
        },
        {
          "id": "bc69d50c-4bdc-4fea",
          "sentence": "Have lost so much money"
        }
      ]
    }
  ]
}
```

## exec_rebuild_lambda.sh
This shell could rebuild the lambda image on local after you modify the code, and push to ECR, update the AWS lambda without rerun the whole Cloudformation. 
It will be helpful when debugging.



