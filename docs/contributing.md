

# Contributing to ```construct-tracker```


- Install python 3.10
- pip install pre-commit
```
pip install pre-commit
nano ~/.zshrc
```
  - add line: `export PATH="$PATH:$HOME/.local/bin"`
  - reload: `source ~/.zshrc`

- Install poetry
```
curl -sSL https://install.python-poetry.org | python3 -
$HOME/.local/bin # Add poetry to PATH
nano ~/.zshrc
```
  - add line: `export PATH="$PATH:$HOME/.local/bin"`
  - reload: `source ~/.zshrc`
  - `poetry --version`

```

poetry lock
poetry install
pre-commit run --all-files
```
Now you can build package locally before committing. This will create a .tar.gz and .whl file for your package in the dist/ directory:
- `poetry build`


Install the Built Package Locally in a virtual environment:

Install miniconda: https://docs.anaconda.com/miniconda/
```
conda create --name ct python=3.10
pip install dist/construct_tracker-1.0.0b0-py3-none-any.whl
```



**Pull requests** are always welcome, and we appreciate any help you give.
Note that a code of conduct applies to all spaces managed by the `construct-tracker` project, including issues and pull requests. Please see the [Code of Conduct](CODE_OF_CONDUCT.md) for details.

## Workflow
Please use the following workflow when contributing:

# Create a virtual environment

```
conda create --name py310 python=3.10.14
conda activate py310
```



0. **Install poetry and versioneer for dynamic versioning**:
  - ```pip install poetry==1.7.1```
  - ```pip install versioneer```
1. **Create an issue**: Use GitHub to create an issue, assign it to yourself (and any collaborators)
2. **Create a branch**: Use GitHub's "Create a branch" button from the issue page to generate a branch associated with the issue.
3. **Clone the repo locally**:
   ```git clone https://github.com/danielmlow/construct-tracker.git```
4. **Checkout locally**:
    - ```git fetch origin```
    - ```git checkout <branch-name>```
5. **Install all required dependencies**:
  - ```poetry run pip install iso-639```
  - ```poetry install --with dev,docs```
6. **Install pre-commit hooks**:
  ```poetry run pre-commit install```
7. **Work locally on the issue branch.**
  Note: The contributed code will be licensed under the same [license](LICENSE) as the rest of the repository. **If you did not write the code yourself, you must ensure the existing license is compatible and include the license information in the contributed files, or obtain permission from the original author to relicense the contributed code.**
8. **Commit and push regularly on your dev branch.**
    - It is also OK to submit work in progress.
    - Please, write unit tests for your code and test it locally:
        ```poetry run pytest```
    - Please, document your code following [Google style guidelines](https://google.github.io/styleguide/) and the example at the end of this document.
      You can manually check the documentation automatically generated from the docstrings:
      ```poetry run pdoc src/construct_tracker -t docs_style/pdoc-theme --docformat google```.
      This command uses ```pdoc``` to generate the documentation for you and make it accessible through a web interface.
    - If you installed the pre-commit hooks properly, some tests and checks will run, and the commit will succeed if all tests pass. If you prefer to run your tests manually, use the following commands:
	  - If you get errors, you might want to clean pre-commit before trying again:
	  ```
	    pre-commit clean
		rm -rf ~/.cache/pre-commit
		rm -rf /Users/danielmlow/.cache/pre-commit
		pre-commit uninstall
		pre-commit install
		poetry lock
	  ```
	  - For all hooks:

	  ```pre-commit run --all-files```
	  - For individual hooks
		- Static type checks:
			```poetry run mypy .```
		- Code style checks:
			```poetry run ruff check```
			- To automatically fix issues:
			```poetry run ruff check --fix```
		- Spell checking:
			```pre-commit run codespell --all-files```
    - add and commit
      ```
      git add .
      git commit -m 'message'
      git fetch --tags
      git rebase origin/main

      git checkout main
      git pull origin main
      git push origin main --tags
      ```

9. **Add repository secrets**: From your github web interface, add the following repository secrets: ```CODECOV_TOKEN``` (CodeCov), ```HF_TOKEN``` (HuggingFace), ```PYPI_TOKEN``` (Pypi).
10. **Submit a pull request**: Once you are done adding your new amazing functionality, submit a pull request to merge the upstream issue branch into the upstream main.


This approach ensures that tasks, issues, and branches all have names that correspond.
It also facilitates incremental neatly scoped changes since it tends to keep the scope of individual changes narrow.

## Debugging what version poetry is using to install

Poetry can have an error where it is using the wrong python version:
```
$ python --version
Python 3.7.3
```

To fix:

```
rm -rf /Users/danielmlow/.cache/pre-commit
brew install pyenv
pyenv install 3.10
pyenv global 3.10
export PATH="$(pyenv root)/shims:$PATH"
export PATH="$HOME/.pyenv/bin:$PATH"
source ~/.bashrc  # or ~/.zshrc if you're using zsh
python --version
pyenv version
poetry env use python3.10.14
cd /Users/danielmlow/Dropbox\ \(MIT\)/datum/
rm .python-version
echo "3.10.14" > .python-version



```


**If you would like to change this workflow, please use the current process to suggest a change to this document.**




### Add a tutorial
If you feel that the functionality you have added to construct-tracker requires some extra explanation, or you want to share some of the knowledge you obtained during the process, create a tutorial and add to `./tutorials/`


### An example of well documented function following Google-style

````
import statistics
from typing import Dict, List

def calculate_statistics(data: List[float]) -> Dict[str, float]:
    """
    Calculate statistics from a list of numbers.

    Args:
        data (list of float): A list of floating-point numbers.

    Returns:
        dict: A dictionary containing the mean, median, variance, and standard deviation of the input data.

    Raises:
        ValueError: If the input data list is empty.

    Examples:
        >>> calculate_statistics([1, 2, 3, 4, 5])
        {'mean': 3.0, 'median': 3.0, 'variance': 2.0, 'std_dev': 1.4142135623730951}

        >>> calculate_statistics([2.5, 3.5, 4.5, 5.5, 6.5])
        {'mean': 4.5, 'median': 4.5, 'variance': 2.5, 'std_dev': 1.5811388300841898}

    Note:
        This function assumes the input data list is not empty. An empty list will raise a ValueError.

    Todo:
        More statistics will be implemented in the future.
    """
    if not data:
        raise ValueError("The input data list is empty.")

    mean = statistics.mean(data)
    median = statistics.median(data)
    variance = statistics.variance(data)
    std_dev = statistics.stdev(data)

    return {
        'mean': mean,
        'median': median,
        'variance': variance,
        'std_dev': std_dev
    }
````


Instructions adapted from [senselab package](https://github.com/sensein/senselab/blob/main/CONTRIBUTING.md)

# Releasing a new version

- Create an API Token on [Pypi](https://pypi.org/).
- Create a [new release](https://github.com/danielmlow/construct-tracker/releases/new) on Github.
Create a new tag in the form ``*.*.*``.


`pyproject.toml` has the requirements

<!-- tutorial to create package: https://www.youtube.com/watch?v=2goLiz4vTss -->

```
conda activate construct_poetry #created before
pip install poetry # create file with dependencies
poetry add pdocs
poetry config virtualenvs.in-project true
poetry lock
poetry install
poetry config pypi-token.pypi API_token
poetry build
pip install twine
twine upload dist/*
poetry publish
```

To reflect any new changes with a local installation
```
poetry build
pip install .
```

To reflect any new changes in pypi, change the version number in pyproject.toml
```
poetry build
poetry publish --build
pip install --upgrade construct-tracker
```


Also, instead of modifying the `pyproject.toml` file by hand, you can use the `add` command. To automatically find a suitable version constraint and install the package and sub-dependencies.
```
poetry add spacy
```

### Steps to include dist/ in GitHub PR
- Commented out dist in `.gitignore`
- Pushed `dist/` contents (should include a .whl file) to GitHub
- Referenced .whl from `pip install` command above
- To generate a new .whl in your `dist` folder
- Run `poetry build`


# Pull requests

Daniel is merging into daniels_branch and then merging into main:

`git fetch origin pull/PULL_REQUEST_NUMBER/head:daniels_branch` pull PR, updates your local copy of the remote branches without changing your current working state

`git checkout daniels_branch` switch to branch. Make sure there are no errors. If there is a discrepancy between local and remote file, you can either commit your local files (with `add` and `commit`) or discard your local changes (`git checkout -- pyproject.toml`) or save them for later (`git stash`)

`git branch` check which branch you're on (the one with an * before it)

`git push origin daniels_branch`  push to branch

Then to make further commits, do:

`git checkout daniels_branch` to ensure the current branch is daniels_branch


```
git add .
git commit -m 'message'
git push --set-upstream origin daniels_branch
```

That will push your branch to the remote repository and set the upstream branch. After running this command, your local daniels_branch will be linked to the remote branch, and you can use git push and git pull commands without specifying the remote branch name in the future




## Testing updates and pull requests

The `@daniels_branch` installs that version.

```
!pip install git+https://github.com/danielmlow/construct-tracker.git@daniels_branch#egg=construct_tracker&subdirectory=dist
```


## Track heavier files with git-lfs

```
brew install git-sizer
git lfs track "src/construct_tracker/data/datasets/reddit_27_subreddits/rmhd_27subreddits_1040posts_train.csv"`
```

Check max file size:

`git-sizer`

This will change .gitattributes. Then add, commit, and push.

If you still have the maximum file size appearing in git-sizer, then maybe you pushed it before tracking it. You can remove from history:
```
brew install bfg
bfg --delete-files rmhd_27subreddits_1040posts_train.csv
git reflog expire --expire=now --all && git gc --prune=now --aggressive
git push origin main --force
```

# pdoc

When running

```
poetry run pdoc src/construct_tracker -t docs_style/pdoc-theme --docformat google
```

You might see errors starting with `Warn` then look for the error below. Fix. Sometimes you need to install a package, so include it as a dependency in the toml file. Then run

```
poetry lock
poetry install
```

# pre-commit

This will do some quality control before each commit.

```
poetry add --dev pre-commit
poetry run pre-commit install
```
