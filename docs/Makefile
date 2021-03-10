# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS     ?=
SPHINXBUILD    ?= sphinx-build
SOURCEDIR      = source
BUILDDIR       = build
DATADIR	       = _data
PACKAGESOURCE  = source/tutorials_source/package
PLATFORMSOURCE = source/tutorials_source/platform
DOCKERSOURCE   = source/docker

ZIPOPTS        ?= -qo

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	make download
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -v

download:
	# inspired by https://github.com/pytorch/tutorials/blob/master/Makefile
	echo "start downloading datasets..."
	mkdir -p $(DATADIR)

	mkdir -p $(PLATFORMSOURCE)/data
	mkdir -p $(PACKAGESOURCE)/data

	# pizza dataset
	wget -N https://storage.googleapis.com/datasets_boris/pizzas.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/pizzas.zip  -d $(PLATFORMSOURCE)
	mkdir -p $(PLATFORMSOURCE)/pizzas/margherita
	mkdir -p $(PLATFORMSOURCE)/pizzas/salami
	mv $(PLATFORMSOURCE)/pizzas/margherita*.jpg $(PLATFORMSOURCE)/pizzas/margherita
	mv $(PLATFORMSOURCE)/pizzas/salami*.jpg $(PLATFORMSOURCE)/pizzas/salami

	# sunflowers dataset
	wget -N https://storage.googleapis.com/datasets_boris/Sunflowers.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/pizzas.zip  -d $(PLATFORMSOURCE)
	mkdir -p $(PLATFORMSOURCE)/pizzas/margherita
	mkdir -p $(PLATFORMSOURCE)/pizzas/salami
	mv $(PLATFORMSOURCE)/pizzas/margherita*.jpg $(PLATFORMSOURCE)/pizzas/margherita
	mv $(PLATFORMSOURCE)/pizzas/salami*.jpg $(PLATFORMSOURCE)/pizzas/salami

	# download images and report for docker
	wget -N https://storage.googleapis.com/datasets_boris/resources.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/resources.zip  -d $(DOCKERSOURCE)

clean-tutorials:
	rm -fr source/tutorials/package
	rm -fr source/tutorials/platform
	rm -fr $(PLATFORMSOURCE)/data
	rm -fr $(PLATFORMSOURCE)/pizzas
	rm -fr $(PLATFORMSOURCE)/lightning_logs
	rm -fr $(DOCKERSOURCE)/resources
	make clean

clean-all: clean-tutorials
	rm -fr _data
