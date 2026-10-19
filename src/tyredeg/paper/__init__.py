"""Analyses that back the paper, kept apart from the core estimation code.

Every number the paper quotes is written by a script in ``scripts/paper/``
into ``paper/results/*.json``. Nothing in the estimation packages
(``data``, ``models``, ``strategy``, ``validation``) imports from here, so a change here
cannot move what they compute.
"""
