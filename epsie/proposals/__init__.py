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
from .normal import (Normal, VeaAdaptiveNormal, SSAdaptiveNormal)
# we'll also promote the Boundaries class to the top-level
from .bounded_normal import (BoundedNormal, SSAdaptiveBoundedNormal, Boundaries,
                             VeaAdaptiveBoundedNormal)
from .angular import (Angular, SSAdaptiveAngular, VeaAdaptiveAngular)
from .discrete import (NormalDiscrete, SSAdaptiveNormalDiscrete,
                       VeaAdaptiveNormalDiscrete,
                       BoundedDiscrete, SSAdaptiveBoundedDiscrete,
                       VeaAdaptiveBoundedDiscrete)


proposals = {
    JointProposal.name: JointProposal,
    Normal.name: Normal,
    SSAdaptiveNormal.name: SSAdaptiveNormal,
    VeaAdaptiveNormal.name: VeaAdaptiveNormal,
    BoundedNormal.name: BoundedNormal,
    SSAdaptiveBoundedNormal.name: SSAdaptiveBoundedNormal,
    VeaAdaptiveBoundedNormal.name: VeaAdaptiveBoundedNormal,
    Angular.name: Angular,
    SSAdaptiveAngular.name: SSAdaptiveAngular,
    VeaAdaptiveAngular.name: VeaAdaptiveAngular,
    NormalDiscrete.name: NormalDiscrete,
    SSAdaptiveNormalDiscrete.name: SSAdaptiveNormalDiscrete,
    VeaAdaptiveNormalDiscrete.name: VeaAdaptiveNormalDiscrete,
    BoundedDiscrete.name: BoundedDiscrete,
    SSAdaptiveBoundedDiscrete.name: SSAdaptiveBoundedDiscrete,
    VeaAdaptiveBoundedDiscrete.name: VeaAdaptiveBoundedDiscrete,
}
