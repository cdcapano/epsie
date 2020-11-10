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
from .normal import (Normal, AdaptiveNormal, SSAdaptiveNormal,
                     ATAdaptiveNormal)
from .eigenvector import (Eigenvector, AdaptiveEigenvector)
from .bounded_eigenvector import (BoundedEigenvector,
                                  AdaptiveBoundedEigenvector)
# we'll also promote the Boundaries class to the top-level
from .bounded_normal import (BoundedNormal, SSAdaptiveBoundedNormal,
                             Boundaries, AdaptiveBoundedNormal,
                             ATAdaptiveBoundedNormal)
from .angular import (Angular, SSAdaptiveAngular, AdaptiveAngular,
                      ATAdaptiveAngular)
from .discrete import (NormalDiscrete, SSAdaptiveNormalDiscrete,
                       AdaptiveNormalDiscrete,
                       BoundedDiscrete, SSAdaptiveBoundedDiscrete,
                       AdaptiveBoundedDiscrete)
from .nested_transdimensional import NestedTransdimensional
from .solid_angle import (IsotropicSolidAngle, AdaptiveIsotropicSolidAngle)
from .birth import (UniformBirth, NormalBirth, LogNormalBirth)


proposals = {
    JointProposal.name: JointProposal,
    Normal.name: Normal,
    Eigenvector.name: Eigenvector,
    BoundedEigenvector.name: BoundedEigenvector,
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
    NestedTransdimensional.name: NestedTransdimensional,
    ATAdaptiveNormal.name: ATAdaptiveNormal,
    ATAdaptiveBoundedNormal.name: ATAdaptiveBoundedNormal,
    ATAdaptiveAngular.name: ATAdaptiveAngular,
    AdaptiveEigenvector.name: AdaptiveEigenvector,
    IsotropicSolidAngle.name: IsotropicSolidAngle,
    AdaptiveIsotropicSolidAngle.name: AdaptiveIsotropicSolidAngle,
}

births = {
    UniformBirth.name: UniformBirth,
    NormalBirth.name: NormalBirth,
    LogNormalBirth.name: LogNormalBirth,
}
