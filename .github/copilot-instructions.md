# GitHub Copilot Development Guidelines

This document defines both the code style conventions and development philosophy for this weather downscaling project.

---

## Part 1: Development Philosophy

### 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- Use docs.
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

### 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

### 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

### 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

## Part 2: Code Style & Conventions

This project maintains consistent code style across PyTorch-based weather downscaling models.

### License Headers

**Every Python file must start with the NVIDIA Apache 2.0 license header:**

```python
# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
```

Update the year range if modifying existing files.

### Naming Conventions

- **Functions & variables**: `snake_case`
- **Classes**: `PascalCase` 
- **Constants**: `UPPER_CASE` (e.g., `EPS = 1e-6`)
- **Private methods**: Prefix with single underscore `_method_name()`
- **Class variables**: Use descriptive names that indicate data type (e.g., `self.stats_path`, `self.dynamic_inputs`)

### Type Hints

**Use comprehensive type hints for function signatures and class attributes:**

```python
def tensor_stats(
    self, variables: Sequence[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Description of what this returns."""
    pass

class ZarrNormalizationStats:
    def __init__(self, stats_path: str | Path):
        self.stats_path: Path = Path(stats_path)
```

Use modern Python 3.10+ union syntax (`str | Path` instead of `Union[str, Path]`) and built-in types (`list`, `dict`, `tuple`) instead of `List`, `Dict`, `Tuple` from `typing`.

### Imports

**Organize imports into three groups, separated by blank lines:**

1. Standard library (`os`, `time`, `logging`, etc.)
2. Third-party (`torch`, `numpy`, `hydra`, etc.)
3. Local/project imports

```python
import os
import time
import logging

import torch
import numpy as np
from omegaconf import DictConfig

from datasets.base import DownscalingDataset
from helpers.train_helpers import set_seed
```

**Import conventions:**
- Use `from module import Symbol` for frequently-used symbols
- Use `import module` for namespaces (especially for large libraries like `torch`, `numpy`)
- Use `logger = logging.getLogger(__file__)` for module-level logging

### Classes & Abstract Base Classes

**For extensible dataset/model classes, use ABC pattern:**

```python
from abc import ABC, abstractmethod

class DownscalingDataset(torch.utils.data.Dataset, ABC):
    """Brief description of the abstract dataset."""
    
    @abstractmethod
    def longitude(self) -> np.ndarray:
        """Get longitude values from the dataset."""
        pass
    
    @abstractmethod
    def output_channels(self) -> List[ChannelMetadata]:
        """Metadata for output channels."""
        pass
```

### Dataclasses

**Use `@dataclass` for data containers and configuration holders:**

```python
from dataclasses import dataclass

@dataclass
class ChannelMetadata:
    """Metadata describing a data channel."""
    name: str
    level: str = ""
    auxiliary: bool = False

@dataclass(frozen=True)
class VariableGroups:
    """Use frozen=True for immutable configurations."""
    dynamic_inputs: list[str]
    static_inputs: list[str]
    targets: list[str]
```

### Docstrings

**Use docstrings on all classes and public methods:**

```python
def normalize(x, means, stds):
    """Normalize data to zero mean and unit variance.
    
    Args:
        x: Input array to normalize
        means: Mean values per channel
        stds: Standard deviation per channel
        
    Returns:
        Normalized array
    """
    x = (x - means) / stds
    return x

class ZarrNormalizationStats:
    """Load and manage normalization statistics from Zarr stores.
    
    This class handles extracting per-variable mean and std statistics
    and providing them as PyTorch tensors.
    """
```

### Comments

**Write comments that explain "why", not "what":**

```python
# Good: explains the rationale
if patch_shape_x != patch_shape_y:
    # Rectangular patches still experimental; square patches only for now
    raise NotImplementedError("Rectangular patch not supported yet")

# Avoid: restates code
# Set batch size per GPU to total batch size divided by world size
batch_gpu_total = total_batch_size // world_size
```

**Use inline comments sparingly** for non-obvious logic:
- Complex mathematical operations
- Non-standard PyTorch patterns
- Workarounds for known issues

### Error Handling

**Raise informative errors with context:**

```python
if not dynamic_inputs:
    raise ValueError(
        "No dynamic predictor variables found. Expected 3D variables starting with 'x_'."
    )

if mean_key not in self.stats:
    raise ValueError(
        f"Could not infer mean/std format for variable '{variable}' in {self.stats_path}."
    )
```

Avoid silent failures; provide enough information for debugging.

### Function/Method Organization

**Order methods in classes logically:**

1. `__init__()` and class setup
2. Abstract methods (if ABC)
3. Public methods (in order of typical usage)
4. Private/helper methods (prefixed with `_`)
5. Magic methods (`__str__`, `__repr__`, etc.)

### Tensor/Array Handling

**Be explicit about tensor vs. NumPy handling:**

```python
def denormalize(x, means, stds):
    # If using numpy arrays, convert tensors to numpy first
    if isinstance(x, np.ndarray):
        if isinstance(stds, torch.Tensor):
            stds = stds.cpu().numpy()
        if isinstance(means, torch.Tensor):
            means = means.cpu().numpy()
    
    x = x * stds + means
    return x
```

Convert to consistent type before operations to avoid implicit type changes.

### Module-Level Constants

**Define constants at module level with documentation:**

```python
logger = logging.getLogger(__file__)
EPS = 1e-6  # Small epsilon to prevent division by zero in normalization
```

### Logging

**Use Python's logging module, not print():**

```python
import logging

logger = logging.getLogger(__file__)

# In code:
logger.warning(f"You are using rectangular patches of shape {patch_shape}")
logger.debug(f"Processing variable: {variable}")
```

### Testing Conventions

- Test files go in `src/tests/`
- Use descriptive test names: `test_normalize_with_numpy_array()`
- Group related tests in test classes

---

## Quick Reference

| Element | Style |
|---------|-------|
| License | NVIDIA Apache 2.0 header in every file |
| Functions/vars | `snake_case` |
| Classes | `PascalCase` |
| Constants | `UPPER_CASE` |
| Type hints | Modern Python 3.10+ (`str \| Path`, `list[str]`) |
| Imports | 3-group organization (stdlib, third-party, local) |
| Classes | Use ABC for extensibility; dataclasses for data |
| Docstrings | Present on all public classes/methods |
| Comments | Explain "why", not "what" |
| Arrays | Be explicit about tensor vs. NumPy types |
| Logging | Use `logging` module, not `print()` |
| Changes | Surgical, traced to requirements |
| Code | Simple, minimum viable solution |
