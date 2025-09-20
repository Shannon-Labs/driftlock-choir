"""Network topology and protocol models.

Note: Python reserves the token ``async``; importing a module named ``async``
using ``from .async import ...`` raises a SyntaxError at parse time. We use
``importlib`` to load it safely at runtime.
"""

from .topo import RandomGeometricGraph, TopologyParams
from .mac import MACProtocol, MACParams, PacketType, MACPacket

from importlib import import_module as _import_module
try:  # pragma: no cover - import-time compatibility shim
    _async_mod = _import_module(__name__ + '.async')
    AsynchronousNetwork = _async_mod.AsynchronousNetwork
    AsyncParams = _async_mod.AsyncParams
    NetworkDelayModel = _async_mod.NetworkDelayModel
except Exception:  # Fallback: define dummies to avoid hard import failures
    AsynchronousNetwork = None  # type: ignore
    AsyncParams = None          # type: ignore
    NetworkDelayModel = None    # type: ignore

__all__ = [
    'RandomGeometricGraph', 'TopologyParams',
    'MACProtocol', 'MACParams', 'PacketType', 'MACPacket',
    'AsynchronousNetwork', 'AsyncParams', 'NetworkDelayModel'
]
