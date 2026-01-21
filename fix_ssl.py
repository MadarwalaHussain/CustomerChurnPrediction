"""
SSL Certificate Verification Fix Module
========================================
This module patches SSL verification for development environments.
WARNING: Only use in development/learning environments, NOT in production!

Usage:
    Import this module BEFORE any other imports that make HTTPS requests:
    
    import fix_ssl
    fix_ssl.apply_ssl_fix()
    
    # Now import your other modules
    from langchain_huggingface import ChatHuggingFace
"""

import sys
import ssl
import os
import warnings
import urllib3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context


class NoVerifyHTTPAdapter(HTTPAdapter):
    """Custom HTTP adapter that disables SSL verification."""

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = create_urllib3_context()
        kwargs['ssl_context'].check_hostname = False
        kwargs['ssl_context'].verify_mode = ssl.CERT_NONE
        return super().init_poolmanager(*args, **kwargs)


_ssl_fix_applied = False


def apply_ssl_fix():
    """
    Apply SSL certificate verification bypass.
    This function is idempotent - calling it multiple times is safe.
    
    WARNING: This disables SSL verification globally. Only use in 
    development/learning environments where you trust the network.
    """
    global _ssl_fix_applied

    if _ssl_fix_applied:
        print("SSL fix already applied, skipping...")
        return

    print("Applying SSL certificate verification fix...")

    # 1. Create unverified SSL context
    ssl._create_default_https_context = ssl._create_unverified_context

    # 2. Disable SSL warnings
    warnings.filterwarnings('ignore', category=urllib3.exceptions.InsecureRequestWarning)
    urllib3.disable_warnings()

    # 3. Set environment variables to disable SSL verification
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['SSL_CERT_FILE'] = ''
    os.environ['PYTHONHTTPSVERIFY'] = '0'

    # 4. Monkey patch requests.Session to use NoVerifyHTTPAdapter
    original_session_init = requests.Session.__init__

    def patched_session_init(self, *args, **kwargs):
        original_session_init(self, *args, **kwargs)
        self.verify = False
        self.mount('https://', NoVerifyHTTPAdapter())
        self.mount('http://', NoVerifyHTTPAdapter())

    requests.Session.__init__ = patched_session_init

    _ssl_fix_applied = True
    print("✓ SSL fix applied successfully")


def is_ssl_fix_applied():
    """Check if SSL fix has been applied."""
    return _ssl_fix_applied


# def print_warning():
#     """Print a warning about SSL verification being disabled."""
#     warning_message = """
#     ⚠️  WARNING: SSL CERTIFICATE VERIFICATION IS DISABLED ⚠️
    
#     This is NOT secure and should only be used in:
#     - Development environments
#     - Learning/testing environments
#     - Trusted networks
    
#     NEVER use this in production or when handling sensitive data!
    
#     For production, properly configure SSL certificates:
#     1. Install corporate certificates if behind a firewall
#     2. Use certifi: pip install --upgrade certifi
#     3. Set REQUESTS_CA_BUNDLE to your certificate path
#     """
#     print(warning_message)


# Auto-apply on import if AUTO_APPLY_SSL_FIX environment variable is set
if os.getenv('AUTO_APPLY_SSL_FIX', '').lower() in ('true', '1', 'yes'):
    apply_ssl_fix()
