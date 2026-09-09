"""Private helper layer: shims and dependency-free numeric drop-ins.

Not public API and not re-exported here on purpose — import the submodules
directly (``from panelary._internal._numpy_stats import skew``) so this
package adds zero import cost to ``import panelary``.
"""
