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
from .normal import (Normal, AdaptiveNormal, SSAdaptiveNormal)
# we'll also promote the Boundaries class to the top-level
from .bounded_normal import (BoundedNormal, SSAdaptiveBoundedNormal,
                             Boundaries, AdaptiveBoundedNormal)
from .angular import (Angular, SSAdaptiveAngular, AdaptiveAngular)
from .discrete import (NormalDiscrete, SSAdaptiveNormalDiscrete,
                       AdaptiveNormalDiscrete,
                       BoundedDiscrete, SSAdaptiveBoundedDiscrete,
                       AdaptiveBoundedDiscrete)
from .poisson import Poisson


proposals = {
    JointProposal.name: JointProposal,
    Normal.name: Normal,
    SSAdaptiveNormal.name: SSAdaptiveNormal,
    AdaptiveNormal.name: AdaptiveNormal,
    BoundedNormal.name: BoundedNormal,
    SSAdaptiveBoundedNormal.name: SSAdaptiveBoundedNormal,
    AdaptiveBoundedNormal.name: AdaptiveBoundedNormal,
    Angular.name: Angular,
    SSAdaptiveAngular.name: SSAdaptiveAngular,
    AdaptiveAngular.name: AdaptiveAngular,
    NormalDiscrete.name: NormalDiscrete,
    SSAdaptiveNormalDiscrete.name: SSAdaptiveNormalDiscrete,
    AdaptiveNormalDiscrete.name: AdaptiveNormalDiscrete,
    BoundedDiscrete.name: BoundedDiscrete,
    SSAdaptiveBoundedDiscrete.name: SSAdaptiveBoundedDiscrete,
    AdaptiveBoundedDiscrete.name: AdaptiveBoundedDiscrete,
    Poisson.name: Poisson,
}
