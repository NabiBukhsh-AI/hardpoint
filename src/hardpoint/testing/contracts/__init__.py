"""Parametrised conformance suites, one per port.

A third party proves an implementation conforms by binding a factory to the kit::

    from hardpoint.testing.contracts import vector_index_contract

    test_my_index = vector_index_contract(lambda: MyIndex(...))

An adapter is not "supported" until it passes its kit (ARCHITECTURE.md §23).
"""
