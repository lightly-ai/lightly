# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS	 			?=
SPHINXBUILD				?= sphinx-build
SOURCEDIR	  			= source
BUILDDIR	   			= build
DATADIR		   			= _data
PACKAGESOURCE  			= source/tutorials_source/package
PLATFORMSOURCE 			= source/tutorials_source/platform
DOCKERSOURCE   			= source/docker
GETTING_STARTED_IMAGES 	= source/getting_started/resources


DATADIR_FREIBURG = /datasets/freiburg_groceries_dataset
DATADIR_CLOTHING = /datasets/clothing-dataset
DATADIR_CIFAR10 = /datasets/cifar10
# Missing dataset: Sentinel dataset is private
# Missing dataset: vin big data (https://github.com/vinbigdata-medical/vindr-cxr)

ZIPOPTS		?= -qo

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	make download
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O) -v

# Build the docs without running the python tutorials
html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(SPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)/html"

# Download the datasets and images needed for html-noplot
download-noplot:
	mkdir -p $(GETTING_STARTED_IMAGES)

	# download resources for s3 integration
	wget -N https://storage.googleapis.com/datasets_boris/resources_s3_integration.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/resources_s3_integration.zip  -d $(GETTING_STARTED_IMAGES)

	# download resources for azure integration
	wget -N https://storage.googleapis.com/datasets_boris/resources_azure_integration.zip -P $(DATADIR)
	unzip $(ZIPOPTS) $(DATADIR)/resources_azure_integration.zip -d $(GETTING_STARTED_IMAGES)
	@if [ -d $(GETTING_STARTED_IMAGES)/resources ]; then \
		mv $(GETTING_STARTED_IMAGES)/resources/* $(GETTING_STARTED_IMAGES)/;\
		rm -R $(GETTING_STARTED_IMAGES)/resources/;\
	fi

	# download images and report for docker
	wget -N https://storage.googleapis.com/datasets_boris/resources.zip -P $(DATADIR);\
	unzip $(ZIPOPTS) $(DATADIR)/resources.zip  -d $(DOCKERSOURCE);\

	# pizza dataset
	@if [ ! -d $(PLATFORMSOURCE)/pizzas/salami ]; then \
		wget -N https://storage.googleapis.com/datasets_boris/pizzas.zip -P $(DATADIR);\
		unzip $(ZIPOPTS) $(DATADIR)/pizzas.zip  -d $(PLATFORMSOURCE);\
		mkdir -p $(PLATFORMSOURCE)/pizzas/margherita;\
		mkdir -p $(PLATFORMSOURCE)/pizzas/salami;\
		mv $(PLATFORMSOURCE)/pizzas/margherita*.jpg $(PLATFORMSOURCE)/pizzas/margherita;\
		mv $(PLATFORMSOURCE)/pizzas/salami*.jpg $(PLATFORMSOURCE)/pizzas/salami;\
	fi

	# sunflowers dataset
	@if [ ! -f $(DATADIR)/Sunflowers.zip ]; then \
		wget -N https://storage.googleapis.com/datasets_boris/Sunflowers.zip -P $(DATADIR);\
	fi

# Download also the datasets needed for the tutorials
download: download-noplot
	# inspired by https://github.com/pytorch/tutorials/blob/master/Makefile
	echo "start downloading datasets..."
	mkdir -p $(DATADIR)

	# Freiburg groceries dataset
	@if [ ! -d $(DATADIR_FREIBURG) ]; then \
		wget -N http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz -P $(DATADIR);\
		mkdir -p $(DATADIR_FREIBURG);\
		tar -xf $(DATADIR)/freiburg_groceries_dataset.tar.gz -C $(DATADIR_FREIBURG);\
	fi

	# Clothing dataset
	@if [ ! -d $(DATADIR_CLOTHING) ]; then \
		git clone https://github.com/alexeygrigorev/clothing-dataset $(DATADIR_CLOTHING);\
	fi

	# Cifar10
	@if [ ! -d $(DATADIR_CIFAR10)/train ]; then \
		wget -N https://storage.googleapis.com/datasets_boris/cifar10.tar -P $(DATADIR);\
		mkdir -p $(DATADIR_CIFAR10);\
		tar -xf $(DATADIR)/cifar10.tar -C $(DATADIR_CIFAR10);\
		mv $(DATADIR_CIFAR10)/cifar10/** $(DATADIR_CIFAR10)/;\
		rm -d $(DATADIR_CIFAR10)/cifar10;\
	fi

clean-tutorials:
	rm -fr source/tutorials/package
	rm -fr source/tutorials/platform
	rm -fr $(PLATFORMSOURCE)/lightning_logs
	rm -fr $(PLATFORMSOURCE)/data
	rm -fr $(PLATFORMSOURCE)/pizzas
	rm -fr $(DOCKERSOURCE)/resources
	rm -fr $(GETTING_STARTED_IMAGES)

clean-all: clean-tutorials
	rm -fr $(DATADIR)
	rm -fr build
