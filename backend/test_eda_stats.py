#!/usr/bin/env python3
import requests
import json

def test_eda_api_with_stats():
    print('=== Testing EDA API with Statistical Summary ===')

    try:
        response = requests.get('http://localhost:8000/eda/BTC', params={'source': 'yfinance'})
        if response.status_code == 200:
            data = response.json()
            print('✅ API Response successful')
            print(f'✅ Success: {data.get("success", False)}')
            
            report = data.get('report', {})
            print(f'✅ Report statistics count: {len(report)}')
            
            if report:
                print('\n📊 Sample Statistics:')
                for key, value in list(report.items())[:8]:
                    if isinstance(value, (int, float)):
                        print(f'  {key}: {value}')
                    elif isinstance(value, dict):
                        print(f'  {key}: {value}')
                    else:
                        print(f'  {key}: {str(value)[:50]}...')
            
            # Check volume formatting
            volume_stats = {k: v for k, v in report.items() if 'volume' in k.lower()}
            print(f'\n💰 Volume Statistics (should have 2 decimal places):')
            for key, value in volume_stats.items():
                print(f'  {key}: {value}')
                
            print(f'\n✅ Charts available: {len(data.get("charts", {}))}')
            
        else:
            print(f'❌ API Error: {response.status_code}')
            print(f'❌ Response: {response.text[:200]}')
            
    except Exception as e:
        print(f'❌ Request failed: {e}')

if __name__ == '__main__':
    test_eda_api_with_stats()

