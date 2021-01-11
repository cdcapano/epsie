#!/usr/bin/env python

"""Creates a list of the available proposals for the docs."""

from epsie import proposals

tmplt = "  * :py:class:`~{}.{}`"
with open('../proposal_list.rst', 'w') as fp:
    for p in proposals.proposals:
        # don't print the joint one since it's special
        if p != 'joint':
            prop = proposals.proposals[p]
            print(tmplt.format(prop.__module__, prop.__name__), file=fp)
