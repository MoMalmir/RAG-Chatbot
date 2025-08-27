.PHONY: install test format lint all

install:
	pip install --upgrade pip &&\
	pip install -r requirements.txt

test:
	python -m pytest -vv test/test_*.py 

format:
	black *.py

lint:
	pylint --disable=R,C *.py test/*.py 

all: install lint test