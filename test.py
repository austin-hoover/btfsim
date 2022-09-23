from __future__ import print_function
from btfsim.util.utils import git_url
from btfsim.util.utils import git_revision_hash
from btfsim.util.utils import is_git_clean
from btfsim.util.utils import url_style


commit = git_revision_hash()
repo_url = git_url()

if commit and repo_url and is_git_clean():
    print("Repository is clean.")
    print('Code should be available at {}'
          .format(url_style('{}/-/tree/{}'.format(repo_url, commit))))
else:
    print("WARNING: Code revision is unknown.")