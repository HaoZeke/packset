====================
packset architecture
====================


One writer on loopback. Cards are files. Atoms are LMDB.
milli is a projection. Every agent is an HTTP client.

Contract
--------

- URL: ``PACKSET_URL`` (``INSIDE_MEMORY_URL`` alias)

- Host: ``127.0.0.1`` only. Never ``localhost``.

- Port: ``8761`` (``PACKSET_PORT``)

- Health: ``packsetd ok``

Split
-----

.. table::

    +------------------+---------------------------------------------------+
    | Crate / script   | Role                                              |
    +==================+===================================================+
    | ``packsetd.py``  | writer                                            |
    +------------------+---------------------------------------------------+
    | ``packset-core`` | named fuse/diversify/decay panel, extract filters |
    +------------------+---------------------------------------------------+

Host panel env: ``PACKSET_FUSE`` ``PACKSET_DIVERSIFY`` ``PACKSET_DECAY``.
Default ``borda`` / ``mmr`` / ``off``. Unknown names fail closed.
Cross-encoder second stage: ``PACKSET_RERANK`` or ``/v1/search?rerank=1``,
off by default. Same ``packset-embed --rerank`` the locomo arm measured.

.. table::

    +--------------------+--------------------------------------+
    | ``packset-client`` | HTTP                                 |
    +--------------------+--------------------------------------+
    | ``packset-milli``  | search projection                    |
    +--------------------+--------------------------------------+
    | ``packset_tui``    | Textual DataTable over GET /v1/atoms |
    +--------------------+--------------------------------------+

Clients do not open the LMDB.

Entities, and the one that is a citation
----------------------------------------

An entity is a free-form name and nothing checks it. The exception is an entity
opening ``deed-`` or ``sha256:``, which names a product in a deed store, and there
the shape is checked where the atom is written rather than found later by a
reader. A bare prefix and a separator that would split the entity into two are
both refused; whether the deed exists is a question the pack cannot ask and does
not try to.

The rule lives twice, once on each side of the port: ``packset_core::atom`` holds
``check_entity`` and ``is_accession``, and ``inside_memory.validate_atom`` enforces it
on the writer that runs today. The Rust copy is the one the daemon inherits.

.. code:: console

    $ packset accessions git:github.com/HaoZeke/vissue
    deed-patch-overlay
    sha256:aabbccdd

That is the pack's half of the two checks a citation is worth: pipe it into
``deedar evidence -`` for whether the bytes are intact, or ``deedar current -`` for
whether the thing cited is still the tip.
