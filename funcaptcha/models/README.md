# Trained model weights (optional)

Place trained `.pt` files here to enable the ML upgrades. The solvers load them automatically when present.

| File | Solver | Train with |
|------|--------|------------|
| `siamese_cycle_match.pt` | cycle_match | `python -m funcaptcha.train_cycle_match --images <dir>` |
| `rotnet_rotation.pt` | rotation | `python -m funcaptcha.train_rotation --images <dir>` |
| `count_quantity.pt` | quantity | `python -m funcaptcha.train_quantity` |
| `dice_classifier.pt` | dice_sum | `python -m funcaptcha.train_dice` |

These files are gitignored. See `funcaptcha/README.md` for full training instructions.
