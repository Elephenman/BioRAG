"""Shared utilities for Skill CLI scripts."""

API_URL = "http://127.0.0.1:8765"


def check_engine(api_url: str = API_URL):
    """Check if the BioRAG Engine is running."""
    import requests
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        return resp.status_code == 200
    except requests.ConnectionError:
        return False