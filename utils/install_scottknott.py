import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector  # R vector of strings

utils = rpackages.importr("utils")
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

# R package names
packnames = ["ScottKnottESD"]

# Selectively install what needs to be installed.
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
print(f"packages to install: {names_to_install}")

if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))
