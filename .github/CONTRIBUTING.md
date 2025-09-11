Contributing to PSI
===================

First off, thank you for taking the time to contribute to PSI!

How to contribute
-----------------

The recommended approach for contributing to PSI is to create a fork of the
[main repository](https://github.com/PythonSI/PSI) on
GitHub, clone it locally, and work on a feature branch. Here's how:

1. Create a fork of the [project repository](https://github.com/PythonSI/PSI)
   by clicking the 'Fork' button at the top right of the page. This will create
   a copy of the codebase under your GitHub account. For detailed instructions on
   forking a repository, see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your forked PSI repository from your GitHub account to your local machine:

   ```bash
   $ git clone https://github.com/PythonSI/PSI.git
   $ cd PSI
   ```

3. Install pre-commit hooks to ensure that your code is properly formatted:

   ```bash
   $ pip install pre-commit
   $ pre-commit install
   ```

   This will install the pre-commit hooks that will run on every commit. If the hooks fail, the commit will be aborted and you'll need to fix the issues before committing again.

4. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``main`` branch directly!

5. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

6. Follow [these instructions](https://help.github.com/articles/creating-a-pull-request-from-a-fork)
to create a pull request from your fork. This will send an email to the committers.

(If any of the above seems like magic to you, please look up the
[Git documentation](https://git-scm.com/documentation) on the web, or ask a friend or another contributor for help.)

Pull Request Checklist
----------------------

We recommended that your contribution complies with the
following rules before you submit a pull request:

-  Follow the PEP8 Guidelines which should be handles automatically by pre-commit.

-  If your pull request addresses an issue, please use the pull request title
   to describe the issue and mention the issue number in the pull request description. This will make sure a link back to the original issue is
   created.

-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.

-  When adding additional functionality, provide at least one
   example script in the ``examples/`` folder. Have a look at other
   examples for reference. Examples should demonstrate why the new
   functionality is useful in practice and, if possible, compare it
   to other methods available in POT.

You can also check for common programming errors with the following
tools:

- All lint checks pass. You can run the following command to check:

  ```bash
  $ pre-commit run --all-files
  ```

  This will run the pre-commit checks on all files in the repository.

<!-- - All tests pass. You can run the following command to check:

  ```bash
   $ pytest --durations=20 -v test/ --doctest-modules
  ```   

Bonus points for contributions that include a performance analysis with
a benchmark script and profiling output (please report on the mailing
list or on the GitHub issue). -->
