"""
FunCaptcha / Arkose-style solvers for the four main challenge types.

1. **Cycle match** (2D object matching) — "Match the animal in left and right image"
   using arrow cycling. Use ``cycle_match``.

2. **Rotation** — "Rotate the 3D object to face this way". Use ``rotation``.

3. **Quantity** — "Change the number of objects until it matches". Use ``quantity``.

4. **Dice sum** — "Select the pair of dice whose top sides add up to X". Use ``dice_sum``.

All modules expose pure logic (image in → answer out). Browser actions (slider drag,
arrow clicks, tile clicks) are in ``browser_actions``; wire them with your Selenium/Playwright
driver and selectors for your target site.
"""

from . import browser_actions
from . import cycle_match
from . import rotation
from . import quantity
from . import dice_sum

__all__ = [
    "browser_actions",
    "cycle_match",
    "rotation",
    "quantity",
    "dice_sum",
]
