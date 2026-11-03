"""Training and evaluation for the ratio manager's learned policy.

Wave 4 of ``ServerNotes/designs/ratio-manager.md``; the plan and the order the
pieces go live in are in ``ServerNotes/designs/ratio-ml.md``.

The module order is the dependency order, and also the order to build in:

``data``
    Read the event log, join outcomes to the decisions that made them, split
    by calendar week.
``features``
    The one feature vector every model shares. Decision-time fields only.
``ranker``
    The expected-return ranker and Thompson-sampling bandit. The first thing
    to ship, and the source of the exploration data everything else needs.
``generators``
    Synthetic swarm curves: parametric, autoregressive, diffusion, GAN —
    judged by whether a policy trained on them transfers.
``dt``
    The decision transformer, in its realized-return and expected-return
    variants. Last, not first.

``data``, ``features`` and ``ranker`` need only numpy and run on the services
VM. ``generators``' learned families and ``dt`` need torch and are meant for
the desktop's GPU.
"""

__all__ = ["data", "features", "ranker", "generators", "dt"]
