"""NEW_TASKS 10 sprint 1: lane budgets, work-conserving scheduler, prefetch packages.

Layout of this subpackage:

* ``config``    - machine profile, lane threads/nice/capacity, cache dirs (section 1).
* ``state``     - ``state.json`` contracts: the only thing the resume logic reads.
* ``scheduler`` - the always-on loop: lanes A/B/G/N, backlog, watchdog, util.csv.
* one module per work package of ``packages.yaml``.

Every package writes its artifacts and then records them; nothing is inferred from
the command line, so a half-finished package can never look ``DONE``.
"""
