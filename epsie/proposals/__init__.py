# Copyright (C) 2019  Collin Capano
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.


from __future__ import absolute_import

from .base import BaseProposal
from .joint import JointProposal
from .normal import (Normal, AdaptiveNormal)
# we'll also promote the Boundaries class to the top-level
from .bounded_normal import (BoundedNormal, AdaptiveBoundedNormal, Boundaries)
from .angular import (Angular, AdaptiveAngular)
from .discrete import (NormalDiscrete, AdaptiveNormalDiscrete,
                       BoundedDiscrete, AdaptiveBoundedDiscrete)


proposals = {
    JointProposal.name: JointProposal,
    Normal.name: Normal,
    AdaptiveNormal.name: AdaptiveNormal,
    BoundedNormal.name: BoundedNormal,
    AdaptiveBoundedNormal.name: AdaptiveBoundedNormal,
    Angular.name: Angular,
    AdaptiveAngular.name: AdaptiveAngular,
    NormalDiscrete.name: NormalDiscrete,
    AdaptiveNormalDiscrete.name: AdaptiveNormalDiscrete,
    BoundedDiscrete.name: BoundedDiscrete,
    AdaptiveBoundedDiscrete.name: AdaptiveBoundedDiscrete,
}
