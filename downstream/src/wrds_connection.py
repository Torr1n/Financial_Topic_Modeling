"""
WRDS Connection Context Manager.

This module provides a context manager for WRDS database connections that
properly manages connection lifecycle. The context manager handles two cases:

1. External connection provided: The connection is used but NOT closed on exit,
   allowing the caller to manage the connection lifecycle.

2. No connection provided: A new WRDS connection is created and automatically
   closed when the context exits (including on exceptions).

Example usage:
    # With shared connection (connection is NOT closed)
    shared_conn = wrds.Connection()
    with WRDSConnection(connection=shared_conn) as conn:
        results = conn.raw_sql("SELECT * FROM table")
    # shared_conn is still open here

    # Without shared connection (connection IS closed)
    with WRDSConnection() as conn:
        results = conn.raw_sql("SELECT * FROM table")
    # connection is automatically closed here
"""


class WRDSConnection:
    """
    Context manager for WRDS database connections.

    Manages connection lifecycle by tracking whether the connection was
    created internally or provided externally. Only internally-created
    connections are closed on exit.

    Attributes:
        _external_connection: Connection provided by caller (if any).
        _connection: The active connection (external or created).
        _owns_connection: Whether this context manager created the connection.
    """

    def __init__(self, connection=None):
        """
        Initialize the context manager.

        Args:
            connection: Optional existing WRDS connection. If provided, this
                connection will be used and NOT closed on exit. If None, a
                new connection will be created and closed on exit.
        """
        self._external_connection = connection
        self._connection = None
        self._owns_connection = False

    def __enter__(self):
        """
        Enter the context and return the WRDS connection.

        Returns:
            The WRDS connection (either external or newly created).
        """
        if self._external_connection is not None:
            self._connection = self._external_connection
            self._owns_connection = False
        else:
            # Import wrds lazily to allow mocking in tests
            import wrds
            self._connection = wrds.Connection()
            self._owns_connection = True
        return self._connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the context and close the connection if we created it.

        Only closes the connection if it was created by this context manager,
        not if it was provided externally.

        Args:
            exc_type: Exception type if an exception was raised.
            exc_val: Exception value if an exception was raised.
            exc_tb: Exception traceback if an exception was raised.

        Returns:
            False to propagate any exception that occurred.
        """
        if self._owns_connection and self._connection is not None:
            self._connection.close()
        return False
