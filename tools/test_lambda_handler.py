"""
Test script for Lambda handler function.

This script tests the Lambda handler locally before deployment.
"""

import json
import os
from dotenv import load_dotenv

from lambda_handler import lambda_handler


def test_standalone_analysis():
    """Test standalone analysis via Lambda handler."""
    print("\n" + "="*80)
    print("Testing Standalone Analysis via Lambda Handler")
    print("="*80 + "\n")

    # Load environment variables
    load_dotenv()

    # Sample event with standalone request
    event = {
        'body': json.dumps({
            'surveyTitle': 'Customer Feedback Survey',
            'theme': 'Product Quality',
            'baseline': [
                {'sentence': 'The product quality is excellent', 'id': '1'},
                {'sentence': 'Very satisfied with the build quality', 'id': '2'},
                {'sentence': 'Great materials used in construction', 'id': '3'},
                {'sentence': 'The price is too high for this quality', 'id': '4'},
                {'sentence': 'Not worth the money', 'id': '5'},
                {'sentence': 'Overpriced for what you get', 'id': '6'},
                {'sentence': 'Delivery was very fast', 'id': '7'},
                {'sentence': 'Shipping speed exceeded expectations', 'id': '8'},
                {'sentence': 'Received the package quickly', 'id': '9'},
                {'sentence': 'Customer service was helpful', 'id': '10'}
            ]
        })
    }

    # Invoke handler
    response = lambda_handler(event, None)

    # Print results
    print(f"Status Code: {response['statusCode']}")
    print(f"Headers: {json.dumps(response['headers'], indent=2)}")

    body = json.loads(response['body'])
    print(f"\nResponse Body:")
    print(json.dumps(body, indent=2))

    if response['statusCode'] == 200:
        print(f"\n✓ Standalone analysis completed successfully")
        print(f"  Generated {len(body.get('clusters', []))} clusters")
    else:
        print(f"\n✗ Error occurred: {body}")


def test_comparative_analysis():
    """Test comparative analysis via Lambda handler."""
    print("\n" + "="*80)
    print("Testing Comparative Analysis via Lambda Handler")
    print("="*80 + "\n")

    # Load environment variables
    load_dotenv()

    # Sample event with comparative request
    event = {
        'body': json.dumps({
            'surveyTitle': 'Product Comparison Survey',
            'theme': 'User Experience',
            'baseline': [
                {'sentence': 'The interface is intuitive and easy to use', 'id': 'b1'},
                {'sentence': 'Navigation is straightforward', 'id': 'b2'},
                {'sentence': 'The design is clean and modern', 'id': 'b3'},
                {'sentence': 'Loading times are acceptable', 'id': 'b4'},
                {'sentence': 'Some features are hard to find', 'id': 'b5'},
                {'sentence': 'The search function works well', 'id': 'b6'}
            ],
            'comparison': [
                {'sentence': 'The new interface is much better', 'id': 'c1'},
                {'sentence': 'Navigation has improved significantly', 'id': 'c2'},
                {'sentence': 'Love the updated design', 'id': 'c3'},
                {'sentence': 'Much faster loading times now', 'id': 'c4'},
                {'sentence': 'All features are easier to access', 'id': 'c5'},
                {'sentence': 'Search is more powerful and accurate', 'id': 'c6'}
            ]
        })
    }

    # Invoke handler
    response = lambda_handler(event, None)

    # Print results
    print(f"Status Code: {response['statusCode']}")
    print(f"Headers: {json.dumps(response['headers'], indent=2)}")

    body = json.loads(response['body'])
    print(f"\nResponse Body:")
    print(json.dumps(body, indent=2))

    if response['statusCode'] == 200:
        print(f"\n✓ Comparative analysis completed successfully")
        print(f"  Generated {len(body.get('clusters', []))} clusters")
    else:
        print(f"\n✗ Error occurred: {body}")


def test_error_handling():
    """Test error handling in Lambda handler."""
    print("\n" + "="*80)
    print("Testing Error Handling")
    print("="*80 + "\n")

    # Load environment variables
    load_dotenv()

    # Test 1: Invalid JSON
    print("Test 1: Invalid JSON")
    event = {'body': 'invalid json {'}
    response = lambda_handler(event, None)
    print(f"  Status: {response['statusCode']}")
    body = json.loads(response['body'])
    print(f"  Error: {body.get('error', {}).get('code')}")

    # Test 2: Missing required fields
    print("\nTest 2: Missing required fields")
    event = {'body': json.dumps({'surveyTitle': 'Test'})}
    response = lambda_handler(event, None)
    print(f"  Status: {response['statusCode']}")
    body = json.loads(response['body'])
    print(f"  Error: {body.get('error', {}).get('code')}")

    # Test 3: Empty baseline
    print("\nTest 3: Empty baseline")
    event = {
        'body': json.dumps({
            'surveyTitle': 'Test',
            'theme': 'Test Theme',
            'baseline': []
        })
    }
    response = lambda_handler(event, None)
    print(f"  Status: {response['statusCode']}")
    body = json.loads(response['body'])
    print(f"  Error: {body.get('error', {}).get('code')}")

    print("\n✓ Error handling tests completed")


if __name__ == '__main__':
    # Check if OpenAI API key is set
    load_dotenv()
    if not os.environ.get('OPENAI_API_KEY'):
        print("ERROR: OPENAI_API_KEY not set in environment")
        print("Please create a .env file with your OpenAI API key")
        print("See .env.example for reference")
        exit(1)

    print("\n" + "="*80)
    print("Lambda Handler Test Suite")
    print("="*80)

    # Run tests
    test_error_handling()
    test_standalone_analysis()
    test_comparative_analysis()

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80 + "\n")
