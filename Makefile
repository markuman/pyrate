PYTHON  ?= python3
VENV    ?= .venv
PY      := $(VENV)/bin/python

# install-ziel: eigenes venv (nicht neben der db in ~/.local/share/pyrate!) + symlink nach bin
BINDIR  ?= $(HOME)/.local/bin
APPDIR  ?= $(HOME)/.local/lib/pyrate
CONFIG  ?= $(HOME)/.config/pyrate/config.yaml

.DEFAULT_GOAL := help
.PHONY: help venv test egress egress-real check-appdir install install-link uninstall clean

help: ## zeigt diese hilfe
	@echo "pyrate 🏴‍☠️  – make-targets:"
	@echo
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{ printf "  \033[1mmake %-12s\033[0m %s\n", $$1, $$2 }'
	@echo
	@echo "variables (overridable, e.g. make install BINDIR=/usr/local/bin):"
	@echo "  BINDIR=$(BINDIR)"
	@echo "  APPDIR=$(APPDIR)"

venv: $(VENV)/.ok ## create dev venv in .venv (for make test)

$(VENV)/.ok: pyproject.toml
	@echo "» lege dev-venv in $(VENV) an …"
	@$(PYTHON) -m venv $(VENV)
	@$(PY) -m pip install -q -e '.[dev]'
	@touch $@
	@echo "✓ dev-venv steht"

test: venv ## pytest gegen das lokale fake-backend
	@echo "» pytest …"
	@$(PY) -m pytest -q

egress: venv ## strace-check: nur der backend-host darf kontaktiert werden (fake-backend)
	@echo "» egress-check gegen lokales fake-backend …"
	@$(PY) tests/egress_check.py

egress-real: venv ## egress-check gegen das echte backend aus der config (kostet einen request)
	@echo "» egress-check gegen das backend aus $(CONFIG) …"
	@$(PY) tests/egress_check.py --real

# shared recipe for install/install-link. $(1) = pip target, $(2) = completion note
# achtung: $(call) trennt an kommas, also keine kommas in den argumenten
define do_install
	@echo "» baue venv in $(APPDIR) …"
	@rm -rf $(APPDIR)
	@$(PYTHON) -m venv $(APPDIR)
	@$(APPDIR)/bin/python -m pip install -q --upgrade pip
	@$(APPDIR)/bin/python -m pip install -q $(1)
	@mkdir -p $(BINDIR)
	@ln -sf $(APPDIR)/bin/pyrate $(BINDIR)/pyrate
	@echo "✓ installiert: $(BINDIR)/pyrate → $(APPDIR)/bin/pyrate"
	@$(BINDIR)/pyrate --version
	@case ":$$PATH:" in *":$(BINDIR):"*) ;; \
	  *) echo "⚠ $(BINDIR) ist nicht im PATH. ab in die shell-rc: export PATH=\"$(BINDIR):\$$PATH\"";; esac
	@test -f $(CONFIG) || echo "⚠ noch keine config: $(CONFIG) anlegen (beispiel in README.md)"
	@echo "  $(2)"
endef

# install/uninstall machen rm -rf $(APPDIR): ein tippfehler (APPDIR=/opt) soll nichts fremdes rasieren
check-appdir:
	@case "$(APPDIR)" in */pyrate) ;; \
	  *) echo "✗ APPDIR muss auf /pyrate enden (ist: '$(APPDIR)'), sonst gibt's kein rm -rf"; exit 1;; esac

install: check-appdir ## pyrate nach BINDIR installieren (code wird kopiert – das ist der normalfall)
	$(call do_install,.,pyrate's code is copied: if you change something IN pyrate itself – 'make install' again.)

install-link: check-appdir ## only for pyrate development: links to the repo instead of copying
	$(call do_install,-e $(CURDIR),pyrate's code is linked to $(CURDIR): changes to it are live immediately.)

uninstall: check-appdir ## pyrate aus BINDIR entfernen (config + sessions bleiben)
	@echo "» entferne $(BINDIR)/pyrate und $(APPDIR) …"
	@rm -f $(BINDIR)/pyrate
	@rm -rf $(APPDIR)
	@echo "✓ weg. config (~/.config/pyrate) und sessions (~/.local/share/pyrate) sind unangetastet."

clean: ## delete dev venv, caches and build leftovers
	@echo "» cleaning up …"
	@rm -rf $(VENV) .pytest_cache .ruff_cache *.egg-info build dist
	@find . -name __pycache__ -type d -prune -exec rm -rf {} +
	@echo "✓ sauber"
