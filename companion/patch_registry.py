\"\"\"
Central registry for self-healing patch handlers.
Add new patches here so the system can discover and execute them.
\"\"\"

from companion.patchers.example_patch import example_patch
# If your code_patcher.py defines a function called patch_code, import it here
try:
    from companion.patchers.code_patcher import patch_code
except ImportError:
    patch_code = None

# If your config_patcher.py defines a function called patch_config, import it here
try:
    from companion.patchers.config_patcher import patch_config
except ImportError:
    patch_config = None

# Collect available handlers in a list
PATCH_HANDLERS = []

# Add handlers if they are defined
if example_patch:
    PATCH_HANDLERS.append(example_patch)
if patch_code:
    PATCH_HANDLERS.append(patch_code)
if patch_config:
    PATCH_HANDLERS.append(patch_config)
