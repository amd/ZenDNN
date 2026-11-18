# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
"""Measurement backends.

benchdnn and replay run on pods: jobs.Session writes one job script that runs
many candidates in one pod session (probe runs with the API log, then the
built-in plan and the candidates interleaved and repeated) and parses its
outputs.  synthetic runs on the host from a cost model, for tests and dry runs.
"""

from .benchdnn import BenchdnnBackend
from .replay import ReplayBackend
from .synthetic import SyntheticBackend

BACKENDS = {"benchdnn": BenchdnnBackend, "replay": ReplayBackend, "synthetic": SyntheticBackend}
