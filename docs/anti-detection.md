# Anti-Detection Guide

This document explains every anti-detection layer in the pipeline, **why** it
exists, and **how** it works.  All code lives in two modules:

| Module | Responsibility |
|---|---|
| `pipeline/browser/driver.py` | WebDriver creation (undetected-chromedriver + stealth JS) |
| `pipeline/browser/stealth.py` | Human-like helpers, rate-limit cooldown, driver disconnect |

---

## Table of Contents

1. [Undetected-Chromedriver](#1-undetected-chromedriver)
2. [Stealth Script Injection](#2-stealth-script-injection)
3. [Human-Like Input Simulation](#3-human-like-input-simulation)
4. [Jittered Sleeps](#4-jittered-sleeps)
5. [Modern User-Agent](#5-modern-user-agent)
6. [Rate-Limit Cooldown File](#6-rate-limit-cooldown-file)
7. [ChromeDriver Disconnect / Reconnect](#7-chromedriver-disconnect--reconnect)
8. [Quick Reference: What Each Detection Checks](#8-quick-reference-what-each-detection-checks)
9. [Maintenance Checklist](#9-maintenance-checklist)

---

## 1. Undetected-Chromedriver

**What TikTok detects:**  
Standard Selenium leaves several fingerprints:

- `navigator.webdriver === true`
- `window.cdc_` variables injected by chromedriver
- Specific Chrome DevTools protocol patterns

**What we do:**  
We use the [`undetected-chromedriver`](https://github.com/ultrafunkamsterdam/undetected-chromedriver)
package, which patches the chromedriver binary at startup to remove `cdc_`
variables and modifies the Chrome runtime to hide automation flags.

```python
# pipeline/browser/driver.py
import undetected_chromedriver as uc

driver = uc.Chrome(options=options, headless=headless, use_subprocess=True)
```

**Fallback:**  
If `undetected-chromedriver` is not installed, the pipeline falls back to
standard `selenium.webdriver.Chrome` with manual stealth patches (less
effective but still functional).

**Install:**

```bash
pip install undetected-chromedriver>=3.5.5
```

---

## 2. Stealth Script Injection

**What TikTok detects:**  
Even with undetected-chromedriver, certain browser properties can reveal
automation:

| Property | Normal Browser | Selenium Default |
|---|---|---|
| `navigator.plugins` | Array of 3–5 plugins | Empty array |
| `navigator.languages` | `['en-US', 'en']` | `['']` or undefined |
| `window.chrome.runtime` | Object exists | Missing |
| `navigator.permissions.query` | Returns real values | Throws or returns generic |

**What we do:**  
After driver creation, we inject a comprehensive stealth script via
`Page.addScriptToEvaluateOnNewDocument` (runs on every new page load):

```javascript
// Fake plugins (empty = headless giveaway)
Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });

// Standard languages
Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });

// Chrome runtime (missing in automation)
if (!window.chrome) window.chrome = {};
if (!window.chrome.runtime) window.chrome.runtime = {};

// Permissions API patch
// Platform, vendor, maxTouchPoints consistency
```

**Code:** `pipeline/browser/stealth.py` → `inject_stealth_scripts()`

---

## 3. Human-Like Input Simulation

**What TikTok detects:**  
`element.send_keys("password")` types the entire string in ~1ms.  No human
types that fast.  TikTok's JS monitors keystroke timing.

**What we do:**

### `human_type(element, text)`
Types one character at a time with random delays (50–180ms per character):

```python
for char in text:
    element.send_keys(char)
    time.sleep(random.uniform(0.05, 0.18))  # 50-180ms per char
```

### `human_click(driver, element)`
Moves the mouse to the element with a small random offset (±3px), pauses
briefly (100–400ms), then clicks:

```python
ActionChains(driver)
    .move_to_element_with_offset(element, offset_x, offset_y)
    .pause(random.uniform(0.1, 0.4))
    .click()
    .perform()
```

**Code:** `pipeline/browser/stealth.py` → `human_type()`, `human_click()`

**Used in:** `pipeline/auth/login.py` for all form interactions.

---

## 4. Jittered Sleeps

**What TikTok detects:**  
Perfectly timed pauses (exactly 2.000s, 3.000s) are a strong bot signal.
Real users have variable reaction times.

**What we do:**  
Replace every `time.sleep(n)` with `human_sleep(n, jitter=j)`:

```python
def human_sleep(base: float, jitter: float = 0.5) -> None:
    duration = max(0.1, base + random.uniform(-jitter, jitter))
    time.sleep(duration)
```

| Location | Before | After |
|---|---|---|
| Login form fill | `time.sleep(2)` | `human_sleep(2.0, jitter=1.0)` |
| Between videos | `time.sleep(delay)` | `human_sleep(delay, jitter=1.0)` |
| Feed scrolling | `time.sleep(1.5)` | `human_sleep(1.5, jitter=0.8)` |
| Page loads | `time.sleep(3)` | `human_sleep(3.0, jitter=1.0)` |

**Code:** `pipeline/browser/stealth.py` → `human_sleep()`

**Used everywhere:** `login.py`, `run.py`, `for_you.py`

---

## 5. Modern User-Agent

**What TikTok detects:**  
Outdated Chrome versions (e.g. Chrome/90) that no real user would run.
TikTok cross-references the claimed version with browser capabilities.

**What we do:**  
Ship a recent stable Chrome user-agent in `driver.py`:

```python
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
```

**Maintenance:** Update this string when Chrome releases a new major version
(every ~4 weeks). Check [Chrome Release Schedule](https://chromiumdash.appspot.com/schedule).

---

## 6. Rate-Limit Cooldown File

**What TikTok does:**  
After too many login attempts, TikTok shows *"Maximum number of attempts
reached. Try again later."* and blocks the account for 30–60 minutes.

**What we do:**  
When we detect this message, we write a cooldown timestamp to a local file
(`data/tiktok/.rate_limit_until`). On the next run, the pipeline reads this
file and refuses to start until the cooldown has elapsed.

```python
# Set cooldown (in login.py, on rate-limit detection)
set_rate_limit(minutes=45)

# Check cooldown (in run.py, before starting)
if is_rate_limited():
    logger.warning("Exiting: rate-limit cooldown is active.")
    return 0

# Clear cooldown (in run.py, after successful login)
clear_rate_limit()
```

**File format:** A single Unix timestamp (epoch seconds) representing when the
cooldown expires.

**Code:** `pipeline/browser/stealth.py` → `is_rate_limited()`,
`set_rate_limit()`, `clear_rate_limit()`

---

## 7. ChromeDriver Disconnect / Reconnect

**What TikTok detects:**  
TikTok's JS can probe for an active WebSocket connection to chromedriver
(the DevTools protocol channel). Its presence is a strong automation signal.

**What we do:**  
Right before submitting the login form, we **stop the chromedriver service**
so the browser runs standalone. After submission completes, we reconnect.

```python
# Before submit
disconnect_driver(driver)   # driver.service.stop()
human_sleep(0.3)

# ... click submit ...

# After TikTok processes
human_sleep(5.0)
reconnect_driver(driver)    # driver.service.start()
```

> **Note:** During the disconnect window, any Selenium calls (find_element,
> etc.) will fail. Only use this around fire-and-forget actions like button
> clicks.

**Code:** `pipeline/browser/stealth.py` → `disconnect_driver()`,
`reconnect_driver()`

---

## 8. Quick Reference: What Each Detection Checks

| Detection Method | Signal | Our Counter |
|---|---|---|
| `navigator.webdriver` check | `true` in automation | uc patches to `undefined` |
| `window.cdc_` variables | chromedriver leak | uc removes from binary |
| Empty `navigator.plugins` | Headless flag | Stealth JS fakes array |
| Missing `chrome.runtime` | Automation env | Stealth JS creates stub |
| Keystroke timing analysis | Instant input | `human_type()` 50–180ms/char |
| Mouse movement patterns | Teleporting clicks | `human_click()` with offset |
| Fixed timing intervals | Bot-like cadence | `human_sleep()` with jitter |
| DevTools WebSocket probe | chromedriver channel | `disconnect_driver()` |
| Outdated user-agent | Old Chrome version | Modern UA string (Chrome 131) |
| Rapid login retries | Brute-force pattern | Rate-limit cooldown file |
| Canvas / WebGL fingerprint | Consistent hash | uc prevents headless leak |
| Permissions API anomalies | Generic responses | Stealth JS patches query() |

---

## 9. Maintenance Checklist

Periodically review these items to keep the anti-detection effective:

- [ ] **User-Agent version** — Update `DEFAULT_USER_AGENT` in `driver.py` when
      Chrome ships a new major version (~every 4 weeks).
- [ ] **undetected-chromedriver** — Run `pip install -U undetected-chromedriver`
      to get the latest patches.
- [ ] **Stealth JS** — If TikTok adds new detection checks, add counters to
      `STEALTH_JS` in `stealth.py`.
- [ ] **Selectors** — TikTok changes its DOM frequently. If login breaks,
      update the XPaths in `login.py`.
- [ ] **Cooldown duration** — If TikTok changes its rate-limit window, adjust
      `DEFAULT_COOLDOWN_MINUTES` in `stealth.py`.
- [ ] **Cookie freshness** — Delete `data/tiktok/cookies.json` if sessions
      expire or the account gets flagged.
