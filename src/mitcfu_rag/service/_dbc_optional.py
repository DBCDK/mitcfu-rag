#!/usr/bin/env python3

"""
:mod:`mitcfu_rag.service._dbc_optional` -- optional dbc_pyutils integration

`dbc_pyutils` is only installed via the non-default `dbc` dependency group
(DBC cluster/CI hardware). This module is the single seam through which the
rest of the `service` package touches it, so nothing else needs its own
try/except: `DBC_AVAILABLE` gates every optional feature (`/metrics`,
enriched `/status`, structured JSON logging), and each symbol below is
`None` when the package isn't installed.

Tests simulate "not installed" by monkeypatching `DBC_AVAILABLE` (and the
symbol attributes) on this module, without needing to actually uninstall
`dbc_pyutils`.
"""

try:
    from dbc_pyutils import Statistics
    from dbc_pyutils import build_info
    from dbc_pyutils import create_instance_id
    from dbc_pyutils import install_base_handler
    from dbc_pyutils import metrics_endpoint
    from dbc_pyutils import PrometheusMiddleware
    from dbc_pyutils import setup_logging

    DBC_AVAILABLE = True
except ImportError:
    Statistics = None
    build_info = None
    create_instance_id = None
    install_base_handler = None
    metrics_endpoint = None
    PrometheusMiddleware = None
    setup_logging = None

    DBC_AVAILABLE = False
