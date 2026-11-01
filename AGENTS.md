# Development conventions

Use Julia for maintenance scripts, validation, and numerical processing. Do not run Python or add Python dependencies. Browser JavaScript and native installation/launcher scripts are application components; disclose other necessary exceptions before using them.

The private development root defaults to the sibling `02_PRIVATE_PROJECT_DEVELOPMENT` (or `LUDWIG_PRIVATE_ROOT`). Put all disposable work, caches, generated test fixtures, and temporary scripts inside its `TEMP_DELETEABLE` directory, never beside the repository. Keep useful source/tests, studies, simulation results, references, and reports outside that disposable directory. No important material may depend on its continued existence.
