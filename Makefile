.PHONY: init update-conda init-conda

init: build
	python3.11 -m venv env-therapy
	env-therapy/bin/pip install -r requirements.txt
	env-therapy/bin/python -m ipykernel install --user --name "env-therapy"

init-conda:
	conda create --name env-therapy
	conda env update --name env-therapy --file environment.yml
	conda activate env-mindgames && pip install -r requirements.txt

update-conda:
	git pull
	conda env update --name env-therapy --file environment.yml
	conda activate env-mindgames && pip install -r requirements.txt	
