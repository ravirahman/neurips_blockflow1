from ..interfaces import client_authorizer

_authorized_clients: map(address, bool)
_owner: address

implements: client_authorizer

@public
def __init__():
    self._owner = msg.sender

@public
@constant
def is_client_authorized(client: address) -> bool:
    return self._authorized_clients[client]

@public
def set_client_authorized(client: address):
    assert msg.sender == self._owner, "only owner can authorize clients"
    assert not self._authorized_clients[client]
    self._authorized_clients[client] = True
