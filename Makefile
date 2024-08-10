# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    +=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)


# manual
github_docs:
	rm -rf docs
	mkdir ./docs && touch ./docs/.nojekyll
	@make -C ./ html
	@cp -a ./build/html/. ./docs

# automatic github action push or pull request
github_action_docs:
	rm -rf docs
	mkdir docs && touch docs/.nojekyll
	@cp -a README.rst docsource/README.rst
	rm -rf docsource/_build && mkdir docsource/_build
	rm -rf docsource/_autosummary
	pipx run poetry run sphinx-build -b html docsource docsource/_build/html
	@cp -a docsource/_build/html/* docs