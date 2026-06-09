.PHONY: doctor test compile

doctor:
	python3 agent.py doctor

test:
	PYTHONPATH=src pytest tests -q

compile:
	env PYTHONPYCACHEPREFIX=/private/tmp/jingying_agent_pycache python3 -m compileall agent.py src tests projects/2026-05-fujie-gcard-v1/legacy_scripts
