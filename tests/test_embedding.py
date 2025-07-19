import json
import requests
import os

def test_embedding():
    """Test if the Google embedding API is working with the actual papers"""
    print("=== Testing Google Embedding API ===")
    
    # Load API key
    with open("keys.json") as f:
        api_key = json.load(f)["GOOGLE_API_KEY"]
    
    EMBEDDING_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    
    # Load the papers
    with open("pubmed_dump.jsonl", "r") as f:
        papers = [json.loads(line) for line in f]
    
    print(f"📊 Testing embedding for {len(papers)} papers")
    
    for i, paper in enumerate(papers, 1):
        print(f"\n📄 Testing Paper {i}: {paper['title'][:60]}...")
        
        abstract = paper.get('abstract', '')
        if not abstract:
            print("   ❌ No abstract to embed")
            continue
        
        # Test embedding the first paragraph
        paragraphs = [p.strip() for p in abstract.split('\n') if p.strip()]
        if not paragraphs:
            print("   ❌ No paragraphs found in abstract")
            continue
        
        test_text = paragraphs[0][:1000]  # First 1000 chars of first paragraph
        print(f"   📝 Testing text: {len(test_text)} characters")
        
        try:
            headers = {"Content-Type": "application/json"}
            data = {
                "model": "models/text-embedding-004", 
                "content": {"parts": [{"text": test_text}]}
            }
            
            print("   🔄 Sending request to Google API...")
            response = requests.post(EMBEDDING_API_URL, headers=headers, json=data)
            
            if response.status_code == 200:
                result = response.json()
                embedding = result["embedding"]["values"]
                print(f"   ✅ Success! Embedding length: {len(embedding)} dimensions")
            else:
                print(f"   ❌ API Error: {response.status_code}")
                print(f"   Response: {response.text[:200]}...")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n🎯 Embedding test complete!")

if __name__ == "__main__":
    test_embedding() 