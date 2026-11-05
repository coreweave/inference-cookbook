# SPDX-FileCopyrightText: 2026 CoreWeave, Inc.
# SPDX-License-Identifier: Apache-2.0
# SPDX-PackageName: inference-cookbook

"""Run slime's train script with cwtrain's W&B metrics on the run's ``rollout/step`` axis.

With ``WANDB_PROJECT`` set, the launcher starts the Ray job with this file in place of the train
script, which it names in ``CWTRAIN_TRAIN_SCRIPT``. W&B keeps the metric definitions of the
driver's writer only; see :func:`cwtrain.integrations.slime.run_train_script`.
"""

import os
import sys

from cwtrain.integrations.slime import run_train_script

if __name__ == "__main__":
    run_train_script(os.environ["CWTRAIN_TRAIN_SCRIPT"], sys.argv[1:])
