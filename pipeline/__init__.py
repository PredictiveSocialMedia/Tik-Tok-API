"""
TikTok scraping pipeline.

Components:
- auth: Login and session management
- browser: WebDriver setup and lifecycle
- captcha: Detection and routing to solvers (rotation, slider, funcaptcha)
- feed: For You feed iteration
- parser: Video and comment extraction (3NF schema)
- storage: Payload-based persistence (local SQLite, REST-ready interface)
"""

__version__ = "0.1.0"
