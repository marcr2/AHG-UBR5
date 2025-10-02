"""
Network connectivity fix for UBR5 paper scraper.
Handles DNS resolution failures and network timeouts gracefully.
"""

import os
import sys
import time
import requests
import socket
import urllib3
from urllib3.exceptions import NameResolutionError, ConnectTimeoutError
import logging
from typing import Dict, List, Optional, Tuple
import json
from datetime import datetime

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class NetworkConnectivityFix:
    """
    Handles network connectivity issues and provides fallback mechanisms.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.setup_session()
        
    def setup_session(self):
        """Configure requests session with better timeout and retry settings."""
        # Set longer timeouts
        self.session.timeout = (10, 30)  # (connect timeout, read timeout)
        
        # Add retry adapter
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set user agent
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def test_connectivity(self, url: str, timeout: int = 10) -> Tuple[bool, str]:
        """
        Test connectivity to a specific URL.
        
        Args:
            url: URL to test
            timeout: Timeout in seconds
            
        Returns:
            Tuple of (success, error_message)
        """
        try:
            response = self.session.head(url, timeout=timeout, verify=False)
            return True, f"HTTP {response.status_code}"
        except NameResolutionError as e:
            return False, f"DNS resolution failed: {e}"
        except ConnectTimeoutError as e:
            return False, f"Connection timeout: {e}"
        except requests.exceptions.ConnectionError as e:
            return False, f"Connection error: {e}"
        except requests.exceptions.Timeout as e:
            return False, f"Request timeout: {e}"
        except Exception as e:
            return False, f"Unknown error: {e}"
    
    def test_all_endpoints(self) -> Dict[str, Tuple[bool, str]]:
        """Test connectivity to all required endpoints."""
        endpoints = {
            'ncbi_eutils': 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
            'semantic_scholar': 'https://api.semanticscholar.org/graph/v1/paper/search',
            'google_api': 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent',
            'posthog': 'https://us.i.posthog.com/batch/'
        }
        
        results = {}
        for name, url in endpoints.items():
            self.logger.info(f"Testing connectivity to {name}...")
            success, message = self.test_connectivity(url)
            results[name] = (success, message)
            status_icon = "OK" if success else "FAIL"
            self.logger.info(f"{name}: {status_icon} {message}")
            
        return results
    
    def get_alternative_endpoints(self) -> Dict[str, List[str]]:
        """Get alternative endpoints for each service."""
        return {
            'ncbi_eutils': [
                'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi',
                'https://www.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
            ],
            'semantic_scholar': [
                'https://api.semanticscholar.org/graph/v1/paper/search',
                'https://api.semanticscholar.org/v1/paper/search'
            ],
            'google_api': [
                'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent',
                'https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedText'
            ]
        }
    
    def find_working_endpoint(self, service_name: str) -> Optional[str]:
        """Find a working endpoint for a specific service."""
        alternatives = self.get_alternative_endpoints().get(service_name, [])
        
        for endpoint in alternatives:
            success, _ = self.test_connectivity(endpoint)
            if success:
                self.logger.info(f"Found working endpoint for {service_name}: {endpoint}")
                return endpoint
        
        self.logger.warning(f"❌ No working endpoints found for {service_name}")
        return None
    
    def create_offline_config(self) -> Dict:
        """Create configuration for offline mode."""
        return {
            "offline_mode": True,
            "use_cached_data": True,
            "skip_network_requests": True,
            "data_sources": ["local_files", "cached_embeddings"],
            "created_at": datetime.now().isoformat(),
            "note": "Offline mode activated due to network connectivity issues"
        }
    
    def patch_paperscraper(self):
        """Apply patches to paperscraper to handle network issues."""
        try:
            import paperscraper
            from paperscraper.api import send_request
            
            # Store original function
            original_send_request = send_request
            
            def patched_send_request(*args, **kwargs):
                """Patched send_request with better error handling."""
                max_retries = 3
                retry_delay = 2
                
                for attempt in range(max_retries):
                    try:
                        return original_send_request(*args, **kwargs)
                    except (NameResolutionError, ConnectTimeoutError, requests.exceptions.ConnectionError) as e:
                        self.logger.warning(f"Network error on attempt {attempt + 1}/{max_retries}: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                        else:
                            self.logger.error(f"All retry attempts failed for paperscraper request")
                            raise
                    except Exception as e:
                        self.logger.error(f"Unexpected error in paperscraper: {e}")
                        raise
            
            # Apply patch
            paperscraper.api.send_request = patched_send_request
            self.logger.info("✅ Applied paperscraper network fix")
            
        except ImportError:
            self.logger.warning("paperscraper not available for patching")
        except Exception as e:
            self.logger.error(f"Failed to patch paperscraper: {e}")
    
    def disable_posthog(self):
        """Disable PostHog analytics to avoid network issues."""
        try:
            # Set environment variable to disable PostHog
            os.environ['POSTHOG_DISABLED'] = 'true'
            os.environ['DISABLE_POSTHOG'] = 'true'
            
            # Try to disable in paperscraper if possible
            try:
                import paperscraper
                if hasattr(paperscraper, 'posthog'):
                    paperscraper.posthog.disabled = True
            except:
                pass
                
            self.logger.info("Disabled PostHog analytics")
            
        except Exception as e:
            self.logger.error(f"Failed to disable PostHog: {e}")
    
    def create_network_config(self) -> Dict:
        """Create network configuration file."""
        config = {
            "network_settings": {
                "timeout": 30,
                "max_retries": 3,
                "retry_delay": 2,
                "exponential_backoff": True,
                "verify_ssl": False
            },
            "endpoints": {
                "ncbi_eutils": "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                "semantic_scholar": "https://api.semanticscholar.org/graph/v1/paper/search",
                "google_api": "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
            },
            "offline_mode": False,
            "posthog_disabled": True,
            "created_at": datetime.now().isoformat()
        }
        
        # Test endpoints and update config
        connectivity_results = self.test_all_endpoints()
        config["connectivity_status"] = connectivity_results
        
        return config

def apply_network_fixes():
    """Apply all network fixes."""
    fix = NetworkConnectivityFix()
    
    print("🔧 Applying network connectivity fixes...")
    
    # Test connectivity
    print("\n📡 Testing network connectivity...")
    connectivity_results = fix.test_all_endpoints()
    
    # Disable PostHog
    print("\n🚫 Disabling PostHog analytics...")
    fix.disable_posthog()
    
    # Patch paperscraper
    print("\n🔨 Patching paperscraper...")
    fix.patch_paperscraper()
    
    # Create network config
    print("\n📝 Creating network configuration...")
    config = fix.create_network_config()
    
    # Save config
    with open('network_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Network fixes applied successfully!")
    print(f"📄 Configuration saved to: network_config.json")
    
    # Show summary
    print("\n📊 Connectivity Summary:")
    for service, (success, message) in connectivity_results.items():
        status = "✅" if success else "❌"
        print(f"   {status} {service}: {message}")
    
    return config

if __name__ == "__main__":
    apply_network_fixes()
