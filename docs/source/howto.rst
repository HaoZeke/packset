======
How-to
======


Each section is one task. Every command reads ``PACKSET_URL`` from the
environment; ``packset ensure`` prints it.

Name the workspace once
-----------------------

A workspace is the git remote of the directory you stand in, or ``default``.
Set ``PACKSET_WORKSPACE`` to pin one for a shell:

.. code:: console

    $ export PACKSET_WORKSPACE=seat
    $ packset status

Every verb also takes the workspace as its last argument, and the writers
take ``--workspace WS``.

Pin a set
---------

A set is a named slice of a workspace with its own cards. Pinning one scopes
reads and deduplication to it.

.. code:: console

    $ packset pin review
    $ packset pin
    review
    $ packset pin ""

Ask what was live on a date
---------------------------

Search answers live-now. A dated question reads the store, where a closed
claim still sits:

.. code:: console

    $ packset atoms --as-of 2026-08-01T00:00:00Z seat

``GET /v1/search`` with ``as_of`` runs the same window over a ranked answer.

Retire a claim
--------------

The pack tombstones; it does not erase. The record stays readable and stops
being recalled.

.. code:: console

    $ curl -s $PACKSET_URL/v1/atoms/delete -d '{"workspace":"seat","id":"3f9c...","why":"deed-review-2026-09"}'

``why`` names the deed that showed the claim wrong. The seat's ``ljos forget``
does this with the same field.

Follow the citations
--------------------

A claim that names a deed accession (``deed-...`` or ``sha256:...``) is joined
to the deed store by that id and nothing else.

.. code:: console

    $ packset accessions seat | deedar evidence -    # the bytes are intact
    $ packset accessions seat | deedar current -     # still the tip
    $ packset citers deed-file-note seat             # the claims standing on one deed

Choose the panel
----------------

The writer fuses its ballots with a named panel. Set the three slots in the
writer's environment, not in a client:

.. table::

    +-----------------------+--------------------------------------------------------------------------------------------------------------------+-------------+
    | Variable              | Values                                                                                                             | Default     |
    +=======================+====================================================================================================================+=============+
    | ``PACKSET_FUSE``      | ``combmnz``, ``combsum``, ``rrf``, ``borda``, ``dowdall``, ``copeland``, ``schulze``, ``ranked-pairs``, ``kemeny`` | ``combmnz`` |
    +-----------------------+--------------------------------------------------------------------------------------------------------------------+-------------+
    | ``PACKSET_DIVERSIFY`` | ``mmr``, ``dpp``, ``none``                                                                                         | ``mmr``     |
    +-----------------------+--------------------------------------------------------------------------------------------------------------------+-------------+
    | ``PACKSET_DECAY``     | ``off``, ``on``, ``fsrs``                                                                                          | ``off``     |
    +-----------------------+--------------------------------------------------------------------------------------------------------------------+-------------+

``fsrs`` scales a score by the claim's retrievability from its review clock;
``on`` is a fourteen-day half-life on age. :doc:`Explanation <explanation>`
says what each was measured against.

Turn on the dense scorer
------------------------

The dense ballot needs the ``packset-embed`` binary beside ``packsetd`` or on
``PATH``, and a model it can load. Models are cached under
``$XDG_CACHE_HOME/packset/embed`` unless ``PACKSET_EMBED_CACHE`` names another
directory. Without the binary the pack answers from the lexical ballots and
says so in ``/v1/status``.

.. code:: console

    $ cargo install --git https://github.com/leidarljos/packset packset-embed
    $ packset stop && packset ensure
    $ packset status seat | jq .dense

Run the second stage on one question
------------------------------------

The cross-encoder rerank is measured and off by default. Ask for it per
query, or set ``PACKSET_RERANK=1`` on the writer:

.. code:: console

    $ curl -s "$PACKSET_URL/v1/search?workspace=seat&q=fusion&rerank=1" | jq '.rerank'

Export for a handover, import from one
--------------------------------------

.. code:: console

    $ packset export --into bag/data/atoms seat | deedar export --into bag/data/deeds -

Import is one POST per line of the ``.jsonl``; the seat's ``ljos receive``
with ``--import`` does that after checking the bag. Trust rows travel the same way.

Run the retrieval benchmark
---------------------------

.. code:: console

    $ curl -sSLO https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json
    $ cargo run --release -p packset-daemon --example locomo -- locomo10.json

``PACKSET_LOCOMO_CONVERSATIONS=3`` scores three conversations for a quick
run; the numbers are then comparable to each other and not to a full run.
