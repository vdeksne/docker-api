#!/usr/bin/env python3
"""
Test script for predictions.
Run this after starting the ML service to test predictions.
"""
import requests
import json
import sys

BASE_URL = "http://localhost:5000"

def test_health():
    """Test health endpoint."""
    print("Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print(f"  Status: {response.status_code}")
        print(f"  Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_population_prediction():
    """Test population prediction."""
    print("\nTesting population prediction...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/predict/population",
            params={
                "country": "USA",
                "years_ahead": 5,
                "base_year": 2020
            },
            timeout=30
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Country: {data.get('country')}")
            print(f"  Predictions: {json.dumps(data.get('predictions', {}), indent=4)}")
        else:
            print(f"  Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_migration_prediction():
    """Test migration prediction."""
    print("\nTesting migration prediction...")
    try:
        response = requests.post(
            f"{BASE_URL}/api/predict/migration",
            json={
                "countries": ["USA", "MEX", "CAN"],
                "target_year": 2025,
                "base_year": 2020
            },
            timeout=30
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Countries: {data.get('countries')}")
            print(f"  Predictions (first 5):")
            predictions = data.get('predictions', {})
            for i, (key, value) in enumerate(list(predictions.items())[:5]):
                print(f"    {key}: {value:.2f}")
        else:
            print(f"  Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False

def test_indicator_prediction():
    """Test indicator prediction."""
    print("\nTesting indicator prediction (GDP)...")
    try:
        response = requests.get(
            f"{BASE_URL}/api/predict/indicator",
            params={
                "indicator": "NY.GDP.MKTP.CD",
                "countries": "USA,MEX,CAN",
                "target_year": 2025,
                "base_year": 2020
            },
            timeout=30
        )
        print(f"  Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"  Indicator: {data.get('indicator')}")
            print(f"  Countries: {data.get('countries')}")
            print(f"  Predictions:")
            for country, value in data.get('predictions', {}).items():
                print(f"    {country}: {value:,.2f}")
        else:
            print(f"  Error: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("ML Prediction Service Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("Population Prediction", test_population_prediction()))
    results.append(("Migration Prediction", test_migration_prediction()))
    results.append(("Indicator Prediction", test_indicator_prediction()))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\nAll tests passed! ✓")
        return 0
    else:
        print("\nSome tests failed. ✗")
        return 1

if __name__ == "__main__":
    sys.exit(main())

