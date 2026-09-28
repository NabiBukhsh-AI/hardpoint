"""An adapter module whose third-party dependency is not installed.

Resolving it must produce a MissingDependencyError naming the extra, which is
the "missing extra" misconfiguration `hardpoint doctor` is required to catch.
"""

import hardpoint_vendor_sdk_that_does_not_exist  # noqa: F401
