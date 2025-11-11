# E38_Fase_2 documentation!

## Description

Fase Final Avance de Proyecto, Gestion del Proyecto de Machine Learning

## Commands

The Makefile contains the central entry points for common tasks related to this project.

### Syncing data to cloud storage

* `make sync_data_up` will use `gsutil rsync` to recursively sync files in `data/` up to `gs://DVC/data/`.
* `make sync_data_down` will use `gsutil rsync` to recursively sync files in `gs://DVC/data/` to `data/`.


