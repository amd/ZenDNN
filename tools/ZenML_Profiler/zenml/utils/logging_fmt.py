# *******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# *******************************************************************************

import logging

class ColoredFormatter(logging.Formatter):
    """Logging formatter that wraps messages with ANSI colour codes based
    on the log level (yellow=WARNING, red=ERROR/CRITICAL, green=INFO,
    blue=DEBUG)."""

    _LEVEL_COLOURS = {
        logging.WARNING: "\033[93m",
        logging.ERROR: "\033[91m",
        logging.CRITICAL: "\033[91m",
        logging.INFO: "\033[32m",
        logging.DEBUG: "\033[34m",
    }
    _RESET = "\033[0m"

    def format(self, record):
        """Format the record, then wrap the resulting string with ANSI
        colour codes.  Operates on the final output rather than mutating
        record.msg so that %-style args are resolved first and other
        handlers/formatters sharing the same LogRecord are unaffected."""
        formatted = super().format(record)
        colour = self._LEVEL_COLOURS.get(record.levelno)
        if colour:
            return f"{colour}{formatted}{self._RESET}"
        return formatted
